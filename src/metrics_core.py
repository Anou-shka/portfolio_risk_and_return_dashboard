# src/metrics_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd

TRADING_DAYS = 252

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------- helpers ----------
def _to_w(
    weights: Union[Dict[str, float], pd.Series, Iterable[float]], cols: Iterable[str], norm=True
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

def price_to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    prices = prices.sort_index()
    r = np.log(prices / prices.shift(1)) if kind == "log" else prices.pct_change()
    return r.dropna(how="all")

def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    w = _to_w(weights, returns.columns)
    return returns.fillna(0.0).dot(w)

def nav_from_returns(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0)).cumprod()

def ann_return(returns: pd.Series, geometric=True) -> float:
    r = returns.dropna()
    if r.empty: return np.nan
    if geometric:
        total = (1 + r).prod()
        years = len(r) / TRADING_DAYS
        return total ** (1 / years) - 1
    return r.mean() * TRADING_DAYS

def ann_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) <= 1: return np.nan
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS)

def sharpe(returns: pd.Series, rf_annual=0.0) -> float:
    mu, sig = ann_return(returns), ann_vol(returns)
    if not sig or np.isnan(sig): return np.nan
    return (mu - rf_annual) / sig

def beta_alpha(port: pd.Series, bench: pd.Series, rf_annual=0.0) -> Tuple[float, float]:
    rf_d = rf_annual / TRADING_DAYS
    df = pd.concat([port, bench], axis=1, keys=["p", "b"]).dropna()
    if df.empty: return np.nan, np.nan
    y = df["p"] - rf_d
    x = df["b"] - rf_d
    cov = np.cov(y, x, ddof=1)
    varb = cov[1,1]
    if varb == 0: return np.nan, np.nan
    beta = cov[0,1] / varb
    alpha = (y - beta * x).mean() * TRADING_DAYS
    return float(beta), float(alpha)

def tracking_error(port: pd.Series, bench: pd.Series) -> float:
    spread = (pd.concat([port, bench], axis=1).dropna().pipe(lambda d: d.iloc[:,0]-d.iloc[:,1]))
    if spread.empty: return np.nan
    return spread.std(ddof=1) * np.sqrt(TRADING_DAYS)

def information_ratio(port: pd.Series, bench: pd.Series) -> float:
    te = tracking_error(port, bench)
    if not te or np.isnan(te): return np.nan
    return (ann_return(port) - ann_return(bench)) / te

@dataclass
class DrawdownStats:
    max_drawdown: float
    peak_date: Optional[pd.Timestamp]
    trough_date: Optional[pd.Timestamp]
    recovery_date: Optional[pd.Timestamp]

def max_drawdown(returns: pd.Series) -> DrawdownStats:
    r = pd.Series(returns).dropna()
    if r.empty: return DrawdownStats(np.nan, None, None, None)
    nav = nav_from_returns(r)
    peaks = nav.cummax()
    dd = nav / peaks - 1.0
    if dd.empty: return DrawdownStats(np.nan, None, None, None)
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    rec = nav.loc[trough:][nav.loc[trough:] >= peaks.loc[peak]].index.min()
    return DrawdownStats(float(dd.min()), peak, trough, rec if isinstance(rec, pd.Timestamp) else None)

def cov_matrix(returns: pd.DataFrame, shrink: float = 0.0) -> pd.DataFrame:
    S = returns.cov()
    if shrink <= 0: return S
    D = pd.DataFrame(np.diag(np.diag(S)), index=S.index, columns=S.columns)
    return (1 - shrink) * S + shrink * D

def risk_contributions(weights, cov: pd.DataFrame) -> pd.Series:
    w = _to_w(weights, cov.columns)
    Sigw = cov.values @ w.values
    var = float(w.values @ Sigw)
    if var <= 0: return pd.Series(np.nan, index=cov.columns)
    rc = (w.values * Sigw) / var
    return pd.Series(rc, index=cov.columns)

def optimize_min_variance(returns: pd.DataFrame, bounds=(0.0, 1.0)) -> pd.Series:
    if not _HAVE_SCIPY: return pd.Series(np.nan, index=returns.columns)
    S = returns.cov().values; n = len(returns.columns)
    def f(w): return float(w @ S @ w)
    cons = [{"type":"eq","fun":lambda w: np.sum(w) - 1}]
    res = minimize(f, x0=np.repeat(1/n, n), method="SLSQP", bounds=[bounds]*n, constraints=cons)
    x = res.x if res.success else np.full(n, np.nan)
    return pd.Series(x, index=returns.columns)

def optimize_max_sharpe(returns: pd.DataFrame, rf_annual=0.0, bounds=(0.0,1.0)) -> pd.Series:
    if not _HAVE_SCIPY: return pd.Series(np.nan, index=returns.columns)
    mu_d, S = returns.mean().values, returns.cov().values
    rf_d = rf_annual/ TRADING_DAYS; n=len(mu_d)
    def negS(w):
        num = w @ (mu_d - rf_d); den = np.sqrt(w @ S @ w) + 1e-12
        return -num/den
    cons=[{"type":"eq","fun":lambda w: np.sum(w)-1}]
    res = minimize(negS, x0=np.repeat(1/n,n), method="SLSQP", bounds=[bounds]*n, constraints=cons)
    x = res.x if res.success else np.full(n, np.nan)
    return pd.Series(x, index=returns.columns)

def trace_efficient_frontier(returns: pd.DataFrame, n_points=40, bounds=(0.0,1.0)) -> pd.DataFrame:
    if not _HAVE_SCIPY: return pd.DataFrame(columns=["ret_ann","vol_ann","weights"])
    mu_a = returns.mean()*TRADING_DAYS; S = returns.cov()*TRADING_DAYS; n=len(returns.columns)
    grid = np.linspace(mu_a.min(), mu_a.max(), n_points); out=[]
    def var_w(w): return float(w @ S.values @ w)
    for t in grid:
        cons=[{"type":"eq","fun":lambda w,tt=t: float(w @ mu_a.values - tt)},
              {"type":"eq","fun":lambda w: np.sum(w)-1}]
        res=minimize(var_w, x0=np.repeat(1/n,n), method="SLSQP", bounds=[bounds]*n, constraints=cons)
        if res.success:
            w=res.x
            out.append({"ret_ann":float(w @ mu_a.values), "vol_ann":float(np.sqrt(var_w(w))), "weights":pd.Series(w,index=returns.columns)})
    return pd.DataFrame(out)

def mc_var_es(returns: pd.DataFrame, weights, alpha=0.95, horizon_days=1, n_sims=10000, seed=None) -> Tuple[float,float]:
    rng = np.random.default_rng(seed)
    R = returns.dropna(how="any")
    if R.empty: return np.nan, np.nan
    w=_to_w(weights, R.columns); mu=R.mean().values; S=R.cov().values
    sims = rng.multivariate_normal(mu, S, size=(n_sims, horizon_days))
    path = sims.sum(axis=1)              # total over horizon
    port = path @ w.values
    q = np.quantile(port, 1-alpha)       # left tail
    var, es = -float(q), -float(port[port<=q].mean())
    return var, es

def relative_return_series(port: pd.Series, bench: pd.Series) -> pd.Series:
    rp = nav_from_returns(port); rb = nav_from_returns(bench)
    return rp / rb - 1.0

def aggregate_by_metadata(weights: Dict[str,float], metadata: Dict[str,Dict[str,str]], field="sector") -> pd.Series:
    w = pd.Series(weights, dtype=float)
    lab = pd.Series({k: metadata.get(k,{}).get(field,"Unknown") or "Unknown" for k in w.index})
    g = w.groupby(lab).sum(); s=g.sum()
    return (g/s if s else g).sort_values(ascending=False)
