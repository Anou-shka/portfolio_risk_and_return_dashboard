# src/metrics_core.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # trading days per year used for all annualisation

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ─────────────────────────────── helpers ────────────────────────────────────

def _to_w(
    weights: Union[Dict[str, float], pd.Series, Iterable[float]],
    cols: Iterable[str],
    norm: bool = True,
) -> pd.Series:
    cols = list(cols)
    if isinstance(weights, dict):
        w = pd.Series({c: float(weights.get(c, 0.0)) for c in cols}, index=cols, dtype=float)
    elif isinstance(weights, pd.Series):
        w = weights.reindex(cols).fillna(0.0).astype(float)
    else:
        w = pd.Series(list(weights), index=cols, dtype=float)
    if norm and w.sum() != 0:
        w = w / w.sum()
    return w


# ─────────────────────────── basic series ops ────────────────────────────────

def price_to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    prices = prices.sort_index()
    r = np.log(prices / prices.shift(1)) if kind == "log" else prices.pct_change()
    return r.dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    w = _to_w(weights, returns.columns)
    return returns.fillna(0.0).dot(w)


def nav_from_returns(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0)).cumprod()


# ─────────────────────────── return & risk ──────────────────────────────────

def ann_return(returns: pd.Series, geometric: bool = True) -> float:
    """Compounded annualised return: (product(1+r))^(252/N) - 1."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    if geometric:
        total = (1 + r).prod()
        years = len(r) / TRADING_DAYS
        return float(total ** (1 / years) - 1)
    return float(r.mean() * TRADING_DAYS)


def ann_vol(returns: pd.Series) -> float:
    """Annualised volatility: daily_std(ddof=1) * sqrt(252)."""
    r = returns.dropna()
    if len(r) <= 1:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Sharpe = (ann_return - rf_annual) / ann_vol."""
    mu, sig = ann_return(returns), ann_vol(returns)
    if not sig or np.isnan(sig):
        return np.nan
    return float((mu - rf_annual) / sig)


def sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Sortino = (ann_return - rf) / downside_deviation.

    Downside deviation = sqrt(mean(min(r_i - rf_daily, 0)^2)) * sqrt(252),
    computed over ALL observations (not just the negative ones).
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    rf_d = rf_annual / TRADING_DAYS
    below_target = np.minimum(r.values - rf_d, 0.0)
    downside_dev = float(np.sqrt((below_target ** 2).mean())) * np.sqrt(TRADING_DAYS)
    if not downside_dev or np.isnan(downside_dev):
        return np.nan
    return float((ann_return(r) - rf_annual) / downside_dev)


def treynor(port: pd.Series, bench: pd.Series, rf_annual: float = 0.0) -> float:
    """Treynor = (ann_return_port - rf) / beta."""
    b, _ = beta_alpha(port, bench, rf_annual=rf_annual)
    if np.isnan(b) or b == 0:
        return np.nan
    return float((ann_return(port) - rf_annual) / b)


def calmar(returns: pd.Series) -> float:
    """Calmar = ann_return / abs(max_drawdown). Max drawdown stored as negative."""
    dd = max_drawdown(returns).max_drawdown
    if np.isnan(dd) or dd == 0:
        return np.nan
    return float(ann_return(returns) / abs(dd))


def hist_var_cvar(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    """Historical VaR and CVaR (Expected Shortfall) at confidence level alpha.

    Returns positive loss values: VaR = -quantile(1-alpha), CVaR = -mean of tail.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan, np.nan
    q = float(np.quantile(r, 1.0 - alpha))   # left-tail quantile (negative number)
    var = -q                                   # VaR as positive loss
    tail = r[r <= q]
    cvar = -float(tail.mean()) if not tail.empty else var  # CVaR = average tail loss
    return float(var), float(cvar)


def beta_alpha(
    port: pd.Series, bench: pd.Series, rf_annual: float = 0.0
) -> Tuple[float, float]:
    """
    Beta = cov(rp, rb) / var(rb)  [excess returns, but rf cancels in cov/var].
    Jensen's Alpha = ann_return_port - [rf + beta * (ann_return_bench - rf)].
    Alpha computed as mean daily residual * 252 (equivalent formula).
    """
    rf_d = rf_annual / TRADING_DAYS          # daily rf for excess-return computation
    df = pd.concat([port, bench], axis=1, keys=["p", "b"]).dropna()
    if df.empty:
        return np.nan, np.nan
    y = df["p"] - rf_d   # daily excess portfolio return
    x = df["b"] - rf_d   # daily excess benchmark return
    cov = np.cov(y, x, ddof=1)
    varb = cov[1, 1]
    if varb == 0:
        return np.nan, np.nan
    beta_val = float(cov[0, 1] / varb)      # beta = cov(rp_excess, rb_excess) / var(rb_excess)
    alpha_val = float((y - beta_val * x).mean() * TRADING_DAYS)  # Jensen's alpha annualised
    return beta_val, alpha_val


def tracking_error(port: pd.Series, bench: pd.Series) -> float:
    """Tracking Error = std(port_returns - bench_returns) * sqrt(252)."""
    df = pd.concat([port, bench], axis=1).dropna()
    spread = df.iloc[:, 0] - df.iloc[:, 1]
    if spread.empty:
        return np.nan
    return float(spread.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(port: pd.Series, bench: pd.Series) -> float:
    """Information Ratio = (ann_return_port - ann_return_bench) / tracking_error."""
    te = tracking_error(port, bench)
    if not te or np.isnan(te):
        return np.nan
    return float((ann_return(port) - ann_return(bench)) / te)


# ──────────────────────────────── drawdown ──────────────────────────────────

@dataclass
class DrawdownStats:
    max_drawdown: float          # negative: (trough_nav - peak_nav) / peak_nav
    peak_date: Optional[pd.Timestamp]
    trough_date: Optional[pd.Timestamp]
    recovery_date: Optional[pd.Timestamp]


def max_drawdown(returns: pd.Series) -> DrawdownStats:
    """Max Drawdown = (trough - peak) / peak. Returns negative value."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return DrawdownStats(np.nan, None, None, None)
    nav = nav_from_returns(r)
    peaks = nav.cummax()
    dd = nav / peaks - 1.0   # negative where nav is below its prior peak
    if dd.empty:
        return DrawdownStats(np.nan, None, None, None)
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    rec_candidates = nav.loc[trough:][nav.loc[trough:] >= peaks.loc[peak]].index
    rec = rec_candidates.min() if not rec_candidates.empty else None
    return DrawdownStats(
        float(dd.min()),
        peak,
        trough,
        rec if isinstance(rec, pd.Timestamp) else None,
    )


# ────────────────────────── portfolio construction ──────────────────────────

def cov_matrix(returns: pd.DataFrame, shrink: float = 0.0) -> pd.DataFrame:
    S = returns.cov()
    if shrink <= 0:
        return S
    D = pd.DataFrame(np.diag(np.diag(S)), index=S.index, columns=S.columns)
    return (1 - shrink) * S + shrink * D


def risk_contributions(weights, cov: pd.DataFrame) -> pd.Series:
    """Marginal risk contributions: w_i * (Sigma @ w)_i / portfolio_variance."""
    w = _to_w(weights, cov.columns)
    Sigw = cov.values @ w.values
    var = float(w.values @ Sigw)
    if var <= 0:
        return pd.Series(np.nan, index=cov.columns)
    rc = (w.values * Sigw) / var
    return pd.Series(rc, index=cov.columns)


def optimize_min_variance(returns: pd.DataFrame, bounds=(0.0, 1.0)) -> pd.Series:
    if not _HAVE_SCIPY:
        return pd.Series(np.nan, index=returns.columns)
    S = returns.cov().values
    n = len(returns.columns)

    def min_var(w):
        return float(w @ S @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(min_var, x0=np.repeat(1 / n, n), method="SLSQP",
                   bounds=[bounds] * n, constraints=cons)
    x = res.x if res.success else np.full(n, np.nan)
    return pd.Series(x, index=returns.columns)


def optimize_max_sharpe(
    returns: pd.DataFrame, rf_annual: float = 0.0, bounds=(0.0, 1.0)
) -> pd.Series:
    if not _HAVE_SCIPY:
        return pd.Series(np.nan, index=returns.columns)
    mu_d = returns.mean().values
    S = returns.cov().values
    rf_d = rf_annual / TRADING_DAYS
    n = len(mu_d)

    def neg_sharpe(w):
        num = w @ (mu_d - rf_d)
        den = np.sqrt(w @ S @ w) + 1e-12
        return -num / den

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(neg_sharpe, x0=np.repeat(1 / n, n), method="SLSQP",
                   bounds=[bounds] * n, constraints=cons)
    x = res.x if res.success else np.full(n, np.nan)
    return pd.Series(x, index=returns.columns)


def trace_efficient_frontier(
    returns: pd.DataFrame, n_points: int = 40, bounds=(0.0, 1.0), shrink: float = 0.0
) -> pd.DataFrame:
    if not _HAVE_SCIPY:
        return pd.DataFrame(columns=["ret_ann", "vol_ann", "weights"])
    mu_a = returns.mean() * TRADING_DAYS
    S = cov_matrix(returns, shrink=shrink) * TRADING_DAYS
    n = len(returns.columns)
    grid = np.linspace(mu_a.min(), mu_a.max(), n_points)
    out = []

    def var_w(w):
        return float(w @ S.values @ w)

    for t in grid:
        cons = [
            {"type": "eq", "fun": lambda w, tt=t: float(w @ mu_a.values - tt)},
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]
        res = minimize(var_w, x0=np.repeat(1 / n, n), method="SLSQP",
                       bounds=[bounds] * n, constraints=cons)
        if res.success:
            wv = res.x
            out.append({
                "ret_ann": float(wv @ mu_a.values),
                "vol_ann": float(np.sqrt(var_w(wv))),
                "weights": pd.Series(wv, index=returns.columns),
            })
    return pd.DataFrame(out)


def mc_var_es(
    returns: pd.DataFrame,
    weights,
    alpha: float = 0.95,
    horizon_days: int = 1,
    n_sims: int = 10_000,
    seed=None,
) -> Tuple[float, float]:
    """Monte Carlo VaR and Expected Shortfall via multivariate normal simulation."""
    rng = np.random.default_rng(seed)
    R = returns.dropna(how="any")
    if R.empty:
        return np.nan, np.nan
    w = _to_w(weights, R.columns)
    mu = R.mean().values
    S = R.cov().values
    sims = rng.multivariate_normal(mu, S, size=(n_sims, horizon_days))
    path = sims.sum(axis=1)              # total return over horizon
    port = path @ w.values
    q = np.quantile(port, 1 - alpha)    # left-tail quantile
    var = -float(q)
    es = -float(port[port <= q].mean())
    return var, es


# ─────────────────────────── utility metrics ────────────────────────────────

def relative_return_series(port: pd.Series, bench: pd.Series) -> pd.Series:
    """Cumulative relative return: port_NAV / bench_NAV - 1."""
    rp = nav_from_returns(port)
    rb = nav_from_returns(bench)
    return rp / rb - 1.0


def aggregate_by_metadata(
    weights: Dict[str, float],
    metadata: Dict[str, Dict[str, str]],
    field: str = "sector",
) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    lab = pd.Series(
        {k: metadata.get(k, {}).get(field, "Unknown") or "Unknown" for k in w.index}
    )
    g = w.groupby(lab).sum()
    s = g.sum()
    return (g / s if s else g).sort_values(ascending=False)


# ────────────────────────── startup validation ──────────────────────────────

def validate_all_metrics(
    port: pd.Series,
    bench: pd.Series,
    rf_annual: float = 0.0,
    label: str = "",
) -> Dict:
    """Compute every core metric and log results. Called once at app startup."""
    b, a = beta_alpha(port, bench, rf_annual=rf_annual)
    dd = max_drawdown(port)
    var_h, cvar_h = hist_var_cvar(port, alpha=0.95)
    metrics = {
        "ann_return":     ann_return(port),
        "ann_vol":        ann_vol(port),
        "sharpe":         sharpe(port, rf_annual=rf_annual),
        "sortino":        sortino(port, rf_annual=rf_annual),
        "treynor":        treynor(port, bench, rf_annual=rf_annual),
        "calmar":         calmar(port),
        "beta":           b,
        "alpha_jensen":   a,
        "tracking_error": tracking_error(port, bench),
        "info_ratio":     information_ratio(port, bench),
        "max_drawdown":   dd.max_drawdown,
        "var_95_hist":    var_h,
        "cvar_95_hist":   cvar_h,
    }
    tag = f"[validate:{label}]" if label else "[validate]"
    logging.info(f"{tag} === Metrics Validation ===")
    for k, v in metrics.items():
        v_safe = float(v) if v is not None else float("nan")
        if np.isfinite(v_safe):
            logging.info(f"{tag} {k}: {v_safe:.4f}")
        else:
            logging.info(f"{tag} {k}: NaN")
    return metrics
