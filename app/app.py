# app/app.py
from __future__ import annotations

import logging
import os
import re
import sys
import time
import datetime as dt
import hmac
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

# All "today" comparisons use New York time so the date matches the financial
# market day — Render servers run in UTC, which would show yesterday's date
# for users ahead of midnight NY time.
_NY = ZoneInfo("America/New_York")

def _today() -> dt.date:
    """Current date in America/New_York — consistent on local and Render."""
    return dt.datetime.now(_NY).date()

def _now() -> dt.datetime:
    """Current wall-clock time in America/New_York."""
    return dt.datetime.now(_NY)

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

from src.config import DEFAULT_BENCH
from src.metrics_core import (
    align_to_trading_calendar, ann_return, ann_vol, beta_alpha, calmar,
    cov_matrix, hist_var_cvar, information_ratio, max_drawdown, mc_var_es,
    nav_from_returns, optimize_max_sharpe, optimize_min_variance,
    portfolio_returns, price_to_returns, relative_return_series,
    risk_contributions, sharpe, sortino, tracking_error,
    trace_efficient_frontier, treynor, validate_all_metrics, TRADING_DAYS,
)
from src.metrics_active import (
    intraday_session_returns, latest_daily_returns, weighted_return,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
            y = float(hist["Close"].dropna().iloc[-1])
            # Yahoo quotes ^TNX in percent (4.2 = 4.2%); some feeds use ×10 (42.0).
            return (y / 1000.0) if y > 20 else (y / 100.0)
    except Exception:
        pass
    return 0.045


def _close_matrix(df: pd.DataFrame, syms: tuple[str, ...]) -> pd.DataFrame:
    """Extract a (date × symbol) close matrix from any yf.download layout."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).unique()
        if "Close" in lvl0:
            prices = df["Close"]
        else:  # group_by="ticker" layout: (ticker, field)
            prices = df.xs("Close", axis=1, level=1)
    else:
        raw = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        prices = raw.to_frame(name=syms[0]) if isinstance(raw, pd.Series) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=syms[0])
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices.sort_index().dropna(how="all")


def _yf_download(syms: tuple[str, ...], attempts: int = 2, **kw) -> pd.DataFrame:
    """yf.download with one retry — Yahoo intermittently returns empty frames
    to cloud hosts such as Render."""
    for i in range(attempts):
        try:
            df = yf.download(list(syms), progress=False, threads=True,
                             auto_adjust=True, **kw)
            prices = _close_matrix(df, syms)
            if not prices.empty:
                return prices
        except Exception as exc:
            logging.warning("yf.download attempt %d failed: %s", i + 1, exc)
        if i + 1 < attempts:
            time.sleep(1.5)
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Fetching price history…")
def fetch_history(syms: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """3 years of daily adjusted closes ending today. Cached 5 min.

    Returns (raw_prices, prices, returns):
      raw_prices — every calendar row Yahoo returned (crypto has weekends);
                   used for "latest session" returns.
      prices     — aligned to stock trading days, so 252-day annualisation
                   holds; used for all historical analytics.
      returns    — daily simple returns of `prices`.
    """
    end   = _today() + dt.timedelta(days=2)           # end is exclusive; +2 covers UTC-dated crypto bars
    start = _today() - dt.timedelta(days=365 * 3 + 5) # +5 for weekend buffer
    raw = _yf_download(syms, start=start.isoformat(), end=end.isoformat())
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    idx = pd.DatetimeIndex(raw.index)
    raw.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    raw = raw[~raw.index.duplicated(keep="last")]

    prices = align_to_trading_calendar(raw)
    rets   = price_to_returns(prices)
    logging.info(
        "fetch_history: %d ticker(s) | %s → %s",
        len(syms),
        raw.index[0].date(),
        raw.index[-1].date(),
    )
    return raw, prices, rets


@st.cache_data(ttl=50, show_spinner=False)   # < the 60 s live refresh, so each refresh gets new bars
def fetch_intraday(syms: tuple[str, ...]) -> pd.DataFrame:
    """1-minute closes for the last few days (UTC index). Cached 50 s.
    Five days guarantees the latest session of every asset is included,
    even on Mondays and after holidays."""
    m = _yf_download(syms, period="5d", interval="1m")
    if m.empty:
        return m
    idx = pd.DatetimeIndex(m.index)
    m.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return m[~m.index.duplicated(keep="last")]


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

# ── Optional access gate (set APP_ACCESS_CODE to enable) ──────────────────
_ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "").strip()
if _ACCESS_CODE and not st.session_state.get("access_ok"):
    st.markdown('<div class="pp-title">📈 Portfolio Pulse</div>', unsafe_allow_html=True)
    _code = st.text_input("Access code", type="password", key="access_code_input")
    if _code:
        if hmac.compare_digest(_code.strip(), _ACCESS_CODE):
            st.session_state.access_ok = True
            st.rerun()
        st.error("Incorrect access code.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]

if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = list(_DEFAULT_TICKERS)
if "ticker_meta" not in st.session_state:
    st.session_state.ticker_meta = {}
if "last_add_status" not in st.session_state:
    st.session_state.last_add_status = None   # ("ok"|"warn"|"err", msg)
if "metrics_validated" not in st.session_state:
    st.session_state.metrics_validated = False


def _add_ticker() -> None:
    """Button callback: validate and add the typed / selected ticker, then
    clear the inputs (widget state may only be reset inside a callback)."""
    custom = st.session_state.get("custom_ticker_input", "").strip().upper()
    pick = st.session_state.get("catalog_selectbox", _CATALOG_PLACEHOLDER)
    sym = custom or (extract_ticker(pick) if pick != _CATALOG_PLACEHOLDER else "")
    if not sym:
        st.session_state.last_add_status = ("warn", "Select a ticker from the catalog or type one above.")
        return
    if sym in st.session_state.portfolio_tickers:
        st.session_state.last_add_status = ("warn", f"{sym} is already in your portfolio.")
        return
    valid, meta = validate_and_get_info(sym)
    if not valid:
        st.session_state.last_add_status = ("err", f"❌ '{sym}' not found on Yahoo Finance.")
        return
    st.session_state.portfolio_tickers.append(sym)
    st.session_state.ticker_meta[sym] = meta
    st.session_state.custom_ticker_input = ""
    st.session_state.catalog_selectbox = _CATALOG_PLACEHOLDER
    st.session_state.last_add_status = ("ok", f"✅ {meta.get('name', sym)} ({sym}) added!")


def _sync_rf(src: str, dst: str) -> None:
    """Keep the risk-free slider and number box showing the same value."""
    v = float(st.session_state[src])
    if dst == "rf_slider":
        v = min(max(v, 0.0), 10.0)
    st.session_state[dst] = v

# ══════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="pp-title">📈 Portfolio Pulse</div>
<div class="pp-sub">Clean visuals · Quick insights · Actionable risk</div>
<div class="pp-date">{_today():%B %d, %Y}</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Portfolio Pulse")
    st.caption(f"Today: {_today():%B %d, %Y}")

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

        st.button(
            "＋ Add to Portfolio",
            width="stretch",
            type="primary",
            key="btn_add_ticker",
            on_click=_add_ticker,
            help="A typed ticker takes priority over the catalog selection.",
        )

        # Show result of the last add attempt
        if st.session_state.last_add_status:
            kind, msg = st.session_state.last_add_status
            {"ok": st.success, "warn": st.warning}.get(kind, st.error)(msg)
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
                c1, c2 = st.columns([4, 1], vertical_alignment="center")
                c1.markdown(label)
                # tertiary = no box/padding, so the ✕ isn't clipped in the narrow column
                if c2.button("✕", key=f"rm_{sym}", help=f"Remove {sym}", type="tertiary"):
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
            _w_assets = [s for s in chosen if s != bench]
            _w_default = round(1.0 / len(_w_assets), 2) if _w_assets else 0.0
            for c in _w_assets:
                sliders[c] = st.slider(c, 0.0, 1.0, _w_default, 0.01, key=f"w_{c}")

        st.divider()

        # ── Risk-Free Rate ─────────────────────────────────────────────
        st.markdown("**🏦 Risk-Free Rate**", help="Annual risk-free rate used in all ratio calculations.")
        live_rf = fetch_treasury_rate()
        st.caption(f"🏦 Current 10Y Treasury: **{live_rf * 100:.2f}%**")
        if "rf_slider" not in st.session_state:
            _rf0 = round(min(max(live_rf * 100, 0.0), 10.0), 2)
            st.session_state.rf_slider = _rf0
            st.session_state.rf_number_input = _rf0
        st.slider(
            "Annual RF rate (%)",
            min_value=0.0,
            max_value=10.0,
            step=0.05,
            format="%.2f%%",
            key="rf_slider",
            on_change=_sync_rf, args=("rf_slider", "rf_number_input"),
            help=(
                "The risk-free rate is the return of a risk-free investment (typically "
                "the 10-year US Treasury). It is subtracted from portfolio returns when "
                "computing risk-adjusted metrics like Sharpe and Sortino."
            ),
        )
        st.number_input(
            "Or enter precise value (%)",
            min_value=0.0,
            max_value=20.0,
            step=0.01,
            format="%.2f",
            key="rf_number_input",
            on_change=_sync_rf, args=("rf_number_input", "rf_slider"),
        )
        rf = float(st.session_state.rf_number_input) / 100.0   # decimal form for all calculations

    # ── 🔄 Data ───────────────────────────────────────────────────────────
    with st.expander("🔄 Data", expanded=False):
        st.caption("Daily prices are cached for 5 minutes, intraday prices for 1 minute.")
        data_status = st.empty()   # filled once prices are loaded
        if st.button("🔄 Refresh Data Now", width="stretch", key="btn_refresh"):
            st.cache_data.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
#  LOAD & SLICE DATA
# ══════════════════════════════════════════════════════════════════════════
raw_all, prices_all, rets_all = fetch_history(tuple(ALL_SYMS))

if prices_all.empty:
    st.error(
        "No price data could be loaded from Yahoo Finance (it may be rate-limiting "
        "this server — try **🔄 Refresh Data Now** in a minute). "
        f"Symbols attempted: {', '.join(ALL_SYMS)}"
    )
    st.stop()

# Show last updated in sidebar Data section
data_status.markdown(
    f"✅ **Loaded:** {_now():%H:%M:%S} ET  \n"
    f"Most recent daily bar: **{raw_all.index[-1].date()}**"
)

# Auto-populate metadata for any portfolio ticker missing it
for sym in list(st.session_state.portfolio_tickers):
    if sym not in st.session_state.ticker_meta:
        meta = fetch_ticker_meta(sym)
        st.session_state.ticker_meta[sym] = meta

VIEW_SYMS = list(dict.fromkeys(chosen + [bench]))
prices = prices_all.loc[:, [c for c in VIEW_SYMS if c in prices_all.columns]]
rets   = rets_all.loc[:,   [c for c in VIEW_SYMS if c in rets_all.columns]]

# Error if any chosen ticker returned no data; drop it from the analysis
_no_data = [s for s in VIEW_SYMS if s not in prices.columns or prices[s].dropna().empty]
for sym in _no_data:
    st.error(f"⚠️ No data returned for **{sym}**. It may be delisted or invalid.")
if bench in _no_data:
    st.error(f"Benchmark **{bench}** has no data — pick another benchmark.")
    st.stop()
chosen    = [s for s in chosen if s not in _no_data]
VIEW_SYMS = [s for s in VIEW_SYMS if s not in _no_data]
prices    = prices.loc[:, VIEW_SYMS]
rets      = rets.loc[:, VIEW_SYMS]
if not [s for s in chosen if s != bench]:
    st.warning("Select at least one asset (other than the benchmark) with price data.")
    st.stop()

today_ts = prices.index.max()


def window_bounds(choice: str):
    """(start, end, is_today) of the analysis window, as timestamps."""
    if choice == "Today": return today_ts, today_ts, True
    if choice == "1W":    return today_ts - pd.Timedelta(days=7), today_ts, False
    if choice == "1M":    return today_ts - pd.DateOffset(months=1), today_ts, False
    if choice == "6M":    return today_ts - pd.DateOffset(months=6), today_ts, False
    if choice == "1Y":    return today_ts - pd.DateOffset(years=1), today_ts, False
    if choice == "3Y":    return today_ts - pd.DateOffset(years=3), today_ts, False
    if custom_dates and len(custom_dates) == 2:
        a, b = sorted(pd.Timestamp(d) for d in custom_dates)
        return a, b, False
    st.sidebar.caption("Pick both a start and an end date — showing the full history meanwhile.")
    return prices.index.min(), today_ts, False


cut, end_ts, is_today = window_bounds(range_choice)
hist_px = prices.loc[(prices.index >= cut) & (prices.index <= end_ts)]
hist_rt = rets.loc[(rets.index >= cut) & (rets.index <= end_ts)]
if not is_today and len(hist_rt) < 2:
    st.warning("The selected window has fewer than 2 trading days of returns — pick a wider window.")
    st.stop()

# ── Build weights ──────────────────────────────────────────────────────────
_equal_w = normalize_ex_bench({s: 0.0 if s == bench else 1.0 for s in chosen}, bench)
if weight_mode == "Equal-weighted (excl. benchmark)":
    w = _equal_w
elif weight_mode == "Custom Weights":
    w = normalize_ex_bench({k: v for k, v in sliders.items() if k in chosen}, bench)
    if not w.sum():
        st.warning("All custom weights are 0 — using equal weights instead.")
        w = _equal_w
else:
    # "Today" has a single return row, too little to optimise on: use full history.
    _opt_rt = rets if is_today else hist_rt
    cols_opt = [c for c in _opt_rt.columns if c in chosen and c != bench]
    R_opt    = _opt_rt[cols_opt]
    if len(cols_opt) < 2:
        wopt = pd.Series(np.nan, index=cols_opt)
    elif weight_mode == "Max Sharpe":
        wopt = optimize_max_sharpe(R_opt, rf_annual=rf)
    else:
        wopt = optimize_min_variance(R_opt)
    wopt = wopt.dropna().clip(lower=0.0)
    if wopt.sum() > 0:
        w = wopt / wopt.sum()
    else:
        st.warning(f"{weight_mode} optimisation needs ≥ 2 assets with enough history — using equal weights.")
        w = _equal_w

# ── Summary pills ─────────────────────────────────────────────────────────
asset_ct = len([s for s in chosen if s != bench])
st.markdown(
    f"""
    <div class='pp-pills'>
      <span class='pp-pill'>Assets: <b>{asset_ct}</b></span>
      <span class='pp-pill'>Benchmark: <b>{bench}</b></span>
      <span class='pp-pill'>Weights: <b>{weight_mode.split(" (")[0]}</b></span>
      <span class='pp-pill'>Window: <b>{'Latest session' if is_today else f"{hist_rt.index.min().date()} → {hist_rt.index.max().date()}"}</b></span>
      <span class='pp-pill'>RF: <b>{rf*100:.2f}%</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Portfolio & benchmark return series ──────────────────────────────────
port    = portfolio_returns(hist_rt[[c for c in w.index if c in hist_rt.columns]], w)
bench_s = hist_rt[bench]

# ── Optional metrics self-check (set PP_VALIDATE_METRICS=1); logs only ────
if os.environ.get("PP_VALIDATE_METRICS") == "1" and not st.session_state.metrics_validated:
    st.session_state.metrics_validated = True
    try:
        _, _, val_rets = fetch_history(("SPY", "QQQ"))
        if not val_rets.empty and "SPY" in val_rets.columns and "QQQ" in val_rets.columns:
            spy_r = val_rets["SPY"].dropna()
            qqq_r = val_rets["QQQ"].dropna()
            validate_all_metrics(spy_r, qqq_r, rf_annual=0.045, label="SPY_vs_QQQ")
    except Exception as exc:
        logging.warning("Startup validation skipped: %s", exc)

def _color(v: float) -> str:
    if not np.isfinite(v):
        return "var(--muted)"
    return "var(--good)" if v >= 0 else "var(--bad)"


def _fmt_ret(v: float) -> str:
    return f"{v:+.2%}" if np.isfinite(v) else "N/A"


def render_summary_cards(port_ret: float, bench_ret: float, note: str = "") -> None:
    """Portfolio / benchmark / active-return cards for the latest session."""
    active = port_ret - bench_ret if np.isfinite(port_ret) and np.isfinite(bench_ret) else np.nan
    for col, label, val in zip(
        st.columns(3),
        ["Portfolio", bench, "Active (Portfolio − Bench)"],
        [port_ret, bench_ret, active],
    ):
        col.markdown(
            f"""<div class="metric-card">
              <div class="metric-title">{label}</div>
              <div class="metric-value" style="color:{_color(val)}">{_fmt_ret(val)}</div>
              <div class="metric-note">{note}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_latest_daily(daily: pd.DataFrame, syms: List[str], title: str) -> None:
    """Latest-session return of each asset from daily bars + portfolio summary."""
    daily = daily.reindex([s for s in syms if s in daily.index]).dropna(subset=["ret"])
    if daily.empty:
        st.info("Not enough daily price data to compute the latest session return.")
        return
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    port_ret  = weighted_return(daily["ret"], w)
    bench_ret = float(daily.loc[bench, "ret"]) if bench in daily.index else np.nan
    render_summary_cards(port_ret, bench_ret, note="latest session vs prior close")

    assets = list(daily.index)
    for start in range(0, len(assets), 4):
        row = st.columns(4)
        for col, sym in zip(row, assets[start:start + 4]):
            r = daily.loc[sym]
            col.markdown(
                f"""<div class="metric-card">
                  <div class="metric-title">{sym}</div>
                  <div class="metric-value" style="color:{_color(r['ret'])}">{_fmt_ret(r['ret'])}</div>
                  <div class="metric-note">{r['last']:,.2f} · {r['date']:%a %b %d} vs {r['prev_date']:%b %d}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    bar = daily["ret"].sort_values()
    fig = go.Figure(go.Bar(
        x=bar.index, y=bar.values,
        marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in bar.values],
        hovertemplate="%{x}: %{y:+.2%}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".1%")
    nice_fig(fig, title="Latest Session Return by Asset", xlab="Ticker", ylab="Return")
    st.plotly_chart(fig, width="stretch")
    if daily["date"].nunique() > 1:
        st.caption(
            "Assets are on different calendars (crypto trades 24/7), so each one "
            "shows its own latest session — see the dates on each card."
        )


latest_daily = latest_daily_returns(raw_all.loc[:, VIEW_SYMS])

# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════
tab_perf, tab_live, tab_vol, tab_risk, tab_div = st.tabs(
    ["📈 Performance", "⚡ Live", "📉 Volatility", "⚠️ Risk & Frontier", "🔀 Diversification"]
)

# ════════════════════════════  PERFORMANCE  ══════════════════════════════
with tab_perf:
    if is_today:
        render_latest_daily(latest_daily, VIEW_SYMS, "Latest Session Return (vs Prior Close)")
        st.caption("Switch to 1W / 1M / … for full historical metrics.")
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
        st.plotly_chart(fig_nav, width="stretch")
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
            st.plotly_chart(fig_rel, width="stretch")

# ════════════════════════════  LIVE  ══════════════════════════════════════
@st.fragment(run_every=60)
def live_view() -> None:
    """Live tab. As a fragment it re-runs on its own every 60 s (and on its
    own widget changes) without re-running the rest of the dashboard."""
    st.subheader("Live Intraday View")
    st.caption(f"Auto-refreshes every minute · last update {_now():%H:%M:%S} ET")
    win_opt = st.select_slider(
        "Show last…",
        options=["30m", "1h", "2h", "3h", "6h", "Full session"],
        value="Full session",
        help="Slice the intraday chart to the most recent window.",
    )
    _mins = {"30m": 30, "1h": 60, "2h": 120, "3h": 180, "6h": 360}

    intra_close = fetch_intraday(tuple(VIEW_SYMS))
    rets_min, sessions = intraday_session_returns(intra_close, raw_all)

    if rets_min.empty:
        st.info(
            "Intraday (1-minute) data is unavailable right now — Yahoo may be "
            "rate-limiting this server. Showing the latest daily session instead."
        )
        render_latest_daily(latest_daily, VIEW_SYMS, "Latest Session Return (Daily Data)")
    else:
        port_live  = weighted_return(rets_min, w).rename("Portfolio")
        bench_live = rets_min[bench].rename(bench) if bench in rets_min.columns else None
        missing    = [s for s in VIEW_SYMS if s not in rets_min.columns]

        p_now = float(port_live.iloc[-1]) if port_live.notna().any() else np.nan
        b_now = float(bench_live.iloc[-1]) if bench_live is not None else np.nan
        _as_of = rets_min.index.max().tz_convert(_NY)
        render_summary_cards(p_now, b_now, note=f"as of {_as_of:%a %H:%M} ET")

        view = rets_min
        if win_opt in _mins:
            view = rets_min.loc[rets_min.index >= rets_min.index.max() - pd.Timedelta(minutes=_mins[win_opt])]
        chart_df = pd.concat(
            [port_live.reindex(view.index)] + ([bench_live.reindex(view.index)] if bench_live is not None else []),
            axis=1,
        )
        show_assets = st.toggle("Show individual assets", value=False, key="live_assets_toggle")
        if show_assets:
            chart_df = pd.concat(
                [chart_df, view[[c for c in view.columns if c != bench]]], axis=1
            )
        chart_df.index = chart_df.index.tz_convert(_NY)

        fig_live = px.line(
            chart_df,
            labels={"value": "Return vs prior close", "variable": "Series", "index": "Time (ET)"},
        )
        fig_live.update_yaxes(tickformat=".2%")
        nice_fig(fig_live, title=f"Intraday Return vs Prior Close — {len(chart_df):,} pts",
                 xlab="Time (ET)", ylab="Return")
        st.plotly_chart(fig_live, width="stretch")

        _sess = pd.Series(sessions).dt.strftime("%a %b %d")
        _groups = _sess.groupby(_sess).groups
        st.caption(
            "Session per asset: "
            + " · ".join(f"{d}: {', '.join(syms)}" for d, syms in _groups.items())
            + ". Each asset is measured from its own prior daily close; "
              "stocks outside 09:30–16:00 ET hold their last session's return."
        )
        if missing:
            st.caption(f"No intraday data for: {', '.join(missing)} (excluded; weights renormalised).")


with tab_live:
    live_view()

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
            if max_win <= 10:
                st.info("Not enough data for rolling statistics — choose a longer window (≥ 1M).")
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

                    roll_vol_view  = roll_vol.dropna()
                    beta_roll_view = beta_roll.dropna()

                    c1, c2 = st.columns(2)
                    with c1:
                        fv = px.line(
                            roll_vol_view.rename(f"Rolling {win}D Ann. Vol"),
                            labels={"value": "Ann. Volatility", "index": "Date"},
                        )
                        fv.update_yaxes(tickformat=".1%")
                        nice_fig(fv, title=f"Rolling {win}D Annualised Volatility",
                                 xlab="Date", ylab="Ann. Volatility")
                        st.plotly_chart(fv, width="stretch")
                    with c2:
                        fb = px.line(
                            beta_roll_view.rename(f"Rolling {win}D Beta"),
                            labels={"value": "Beta", "index": "Date"},
                        )
                        nice_fig(fb, title=f"Rolling {win}D Beta vs {bench}",
                                 xlab="Date", ylab="Beta")
                        st.plotly_chart(fb, width="stretch")
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
        if R.empty:
            st.info("Select at least one non-benchmark asset.")
        else:
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
                    st.plotly_chart(fig_rc, width="stretch")
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
                st.plotly_chart(fig_vb, width="stretch")

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
                        if bmin * len(cols_sel) > 1 or bmax * len(cols_sel) < 1:
                            st.warning("Weight bounds are infeasible for this many assets "
                                       "(weights must be able to sum to 100%).")
                        ef = trace_efficient_frontier(
                            R, n_points=n_pts, bounds=(bmin, bmax), shrink=shrink
                        )
                    else:
                        ef = pd.DataFrame()

                    if ef.empty:
                        st.caption("Efficient frontier requires ≥ 2 assets and scipy.")
                    else:
                        # Plot reference portfolios with the same expected-return
                        # (arithmetic mean × 252) and covariance (incl. shrinkage) the
                        # frontier uses, so points are comparable with the curve.
                        mu_a  = R.mean() * TRADING_DAYS
                        S_a   = cov_matrix(R, shrink=shrink) * TRADING_DAYS

                        def _pt(name: str, wv: pd.Series) -> dict:
                            wv = wv.reindex(cols_sel).fillna(0.0)
                            return {"name": name,
                                    "ret_ann": float(wv @ mu_a),
                                    "vol_ann": float(np.sqrt(wv @ S_a @ wv))}

                        w_port = w.reindex(cols_sel).fillna(0.0)
                        rows = [_pt("Portfolio", w_port / w_port.sum() if w_port.sum() else w_port),
                                _pt("Equal-Weight", pd.Series(1.0 / len(cols_sel), index=cols_sel))]
                        try:
                            w_ms = optimize_max_sharpe(R, rf_annual=rf, bounds=(bmin, bmax))
                            w_mv = optimize_min_variance(R, bounds=(bmin, bmax))
                        except Exception:
                            w_ms = w_mv = pd.Series(dtype=float)
                        if w_ms.notna().all() and not w_ms.empty:
                            rows.append(_pt("Max Sharpe", w_ms))
                        if w_mv.notna().all() and not w_mv.empty:
                            rows.append(_pt("Min Variance", w_mv))
                        pts = pd.DataFrame(rows)

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
                        fig_ef.update_xaxes(tickformat=".0%")
                        fig_ef.update_yaxes(tickformat=".0%")
                        nice_fig(fig_ef, title="Efficient Frontier (Annualised, mean returns)",
                                 xlab="Volatility", ylab="Expected Return")
                        st.plotly_chart(fig_ef, width="stretch")
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
            st.plotly_chart(fig_corr, width="stretch")
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
            c1.plotly_chart(fig_sec, width="stretch")

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
            c2.plotly_chart(fig_reg, width="stretch")

    # ── Per-asset metrics table ────────────────────────────────────────
    if not is_today:
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
            st.dataframe(disp, width="stretch")
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
      Last loaded: <b>{_now():%H:%M:%S}</b> &nbsp;·&nbsp;
      Portfolio Pulse
    </footer-bar>
    """,
    unsafe_allow_html=True,
)
