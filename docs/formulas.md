# 📘 FORMULAS — Portfolio Pulse

This doc captures the math used across `metrics_core.py` and `metrics_active.py`.
Conventions: daily returns are decimals (e.g., 0.0123 = 1.23%). We assume **252 trading days** per year.

---

## Notation

* Prices: $P_t$
* Simple return: $r_t$
* Log return: $\ell_t$
* Portfolio weights (column vector): $\mathbf{w}$, with $\sum_i w_i = 1$
* Asset return vector at $t$: $\mathbf{r}_t$
* Benchmark return: $r_t^{(b)}$
* Risk-free (annual): $r_f^{(\text{ann})}$; daily $r_f^{(d)} = \dfrac{r_f^{(\text{ann})}}{252}$
* Mean vector (daily): $\boldsymbol{\mu}$; covariance (daily): $\Sigma$

---

## Returns

### Simple & log returns

$$
r_t = \frac{P_t}{P_{t-1}} - 1, \qquad
\ell_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
$$

> In code: `price_to_returns(prices, kind="simple" | "log")`.

### Portfolio daily return

$$
r^{(p)}_t = \mathbf{w}^\top \mathbf{r}_t
$$

> In code: `portfolio_returns(returns, weights)`.

### Cumulative NAV (growth of 1)

$$
\text{NAV}_t = \text{NAV}_0 \prod_{\tau \le t} (1 + r^{(p)}_\tau), \quad \text{NAV}_0=1
$$

> In code: `nav_from_returns(port_returns, start=1.0)`.

---

## Annualization

Let $r_1,\dots,r_n$ be daily returns.

### Annualized return (geometric by default)

$$
\mu_{\text{ann}} =
\begin{cases}
\left(\prod_{t=1}^{n} (1 + r_t)\right)^{\tfrac{252}{n}} - 1 & \text{(geometric)} \\[6pt]
\bar{r}\cdot 252 & \text{(arithmetic)}
\end{cases}
$$

> In code: `ann_return(series, geometric=True)`.

### Annualized volatility (sample std.)

$$
\sigma_{\text{ann}} \;=\; \sqrt{\text{Var}(r_1,\ldots,r_n)} \;\cdot\; \sqrt{252}
$$

> In code: `ann_vol(series)` uses sample std with `ddof=1`.

---

## Risk-adjusted performance

### Sharpe Ratio (annual)

$$
\text{Sharpe} \;=\; \frac{\mu_{\text{ann}} - r_f^{(\text{ann})}}{\sigma_{\text{ann}}}
$$

> In code: `sharpe(returns, rf_annual)`.

### Beta & Alpha (vs. benchmark)

Using **daily excess returns**:

$$
x_t = r_t^{(b)} - r_f^{(d)}, \qquad
y_t = r_t^{(p)} - r_f^{(d)} .
$$

$$
\beta \;=\; \frac{\text{Cov}(y, x)}{\text{Var}(x)}, 
\qquad
\alpha_{\text{ann}} \;=\; \big(\overline{\,y - \beta x\,}\big)\cdot 252 .
$$

> In code: `beta_alpha(port, bench, rf_annual)`.

### Tracking Error (TE) & Information Ratio (IR)

Active daily return: $a_t = r_t^{(p)} - r_t^{(b)}$.

$$
\text{TE}_{\text{ann}} \;=\; \sqrt{\text{Var}(a)} \cdot \sqrt{252},
\qquad
\text{IR} \;=\; \frac{\mu_{\text{ann}}^{(p)} - \mu_{\text{ann}}^{(b)}}{\text{TE}_{\text{ann}}} .
$$

> In code: `tracking_error(port, bench)`, `information_ratio(port, bench)`.

---

## Drawdowns

Let $\text{NAV}_t$ be portfolio NAV and $\text{Peak}_t = \max_{\tau \le t}\text{NAV}_\tau$.

$$
\text{DD}_t \;=\; \frac{\text{NAV}_t}{\text{Peak}_t} - 1,
\qquad
\text{MaxDrawdown} \;=\; \min_t \text{DD}_t .
$$

> In code: `max_drawdown(returns)` returns magnitude and peak/trough/recovery dates.

---

## Covariance, shrinkage, & risk contributions

### Sample covariance

$$
\Sigma \;=\; \text{Cov}(\mathbf{r}) .
$$

### Shrinkage toward diagonal ($\lambda \in [0,1]$)

$$
\Sigma_\lambda \;=\; (1-\lambda)\,\Sigma \;+\; \lambda\,\text{diag}\!\big(\text{diag}(\Sigma)\big) .
$$

> In code: `cov_matrix(returns, shrink=lambda)`.

### Portfolio variance & marginal risk

$$
\sigma_p^2 \;=\; \mathbf{w}^\top \Sigma \mathbf{w},
\qquad
\mathbf{m} \;=\; \Sigma \mathbf{w} .
$$

### Percentage risk contribution of asset $i$

$$
\text{RC}_i \;=\; \frac{w_i\, m_i}{\sigma_p^2}, 
\qquad \sum_i \text{RC}_i = 1 .
$$

> In code: `risk_contributions(weights, cov)`.

---

## Optimization & Efficient Frontier

Let daily mean vector $\boldsymbol{\mu}$ and covariance $\Sigma$. Bounds $w_i \in [L,U]$; budget $\sum_i w_i = 1$.

### Min-Variance

$$
\min_{\mathbf{w}} \ \mathbf{w}^\top \Sigma \mathbf{w}
\quad \text{s.t.} \ \sum_i w_i = 1,\ \ L \le w_i \le U .
$$

> In code: `optimize_min_variance(returns, bounds)`.

### Max-Sharpe (daily rf used internally)

$$
\max_{\mathbf{w}} \ \frac{\mathbf{w}^\top \big(\boldsymbol{\mu}-r_f^{(d)}\mathbf{1}\big)}{\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}}
\quad \text{s.t.} \ \sum_i w_i = 1,\ \ L \le w_i \le U .
$$

> In code: `optimize_max_sharpe(returns, rf_annual, bounds)`.

### Frontier tracing

For a grid of target **annual** returns $R^{(\text{ann})}$, solve:

$$
\min_{\mathbf{w}} \ \mathbf{w}^\top \Sigma_{\text{ann}} \mathbf{w}
\quad \text{s.t.} \ 
\mathbf{w}^\top \boldsymbol{\mu}_{\text{ann}} = R^{(\text{ann})}, \ 
\sum_i w_i = 1,\ 
L \le w_i \le U ,
$$

with $\boldsymbol{\mu}_{\text{ann}} = 252\,\boldsymbol{\mu}$ and $\Sigma_{\text{ann}} = 252\,\Sigma$.

> In code: `trace_efficient_frontier(returns, n_points, bounds)`.

---

## Monte Carlo VaR & ES (parametric)

Simulate **daily** portfolio returns assuming multivariate normal:

1. Estimate $\boldsymbol{\mu}, \Sigma$ from daily returns.
2. Draw $\mathbf{r}^{(\text{sim})}_{1:H} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ for $H$ days.
3. Sum over horizon to get portfolio return:

$$
R^{(\text{sim})} \;=\; \left(\sum_{h=1}^{H} \mathbf{r}^{(\text{sim})}_{h}\right)^\top \mathbf{w}.
$$

4. VaR/ES (loss, positive number) at confidence $\alpha$:

$$
\text{VaR}_\alpha \;=\; -\,Q_{1-\alpha}\!\big(R^{(\text{sim})}\big), 
\qquad
\text{ES}_\alpha \;=\; -\,\mathbb{E}\!\big[R^{(\text{sim})} \mid R^{(\text{sim})} \le Q_{1-\alpha}\big].
$$

> In code: `mc_var_es(returns, weights, alpha, horizon_days, n_sims, seed)`.

---

## Relative performance series

Cumulative relative return vs benchmark using NAVs:

$$
\text{Rel}_t \;=\; \frac{\text{NAV}^{(p)}_t}{\text{NAV}^{(b)}_t} - 1 .
$$

> In code: `relative_return_series(port, bench)`.

---

## Rolling metrics (used in the app)

Given aligned daily series over a rolling window $W$:

* **Rolling vol (annualized)**:

  $$
  \sigma_{\text{ann, roll}}(t) \;=\; \sqrt{\text{Var}\!\big(r^{(p)}_{t-W+1:t}\big)} \cdot \sqrt{252}.
  $$
* **Rolling beta** (portfolio vs. benchmark):

  $$
  \beta_{\text{roll}}(t) \;=\; 
  \frac{\text{Cov}\!\big(r^{(p)}_{t-W+1:t}, \ r^{(b)}_{t-W+1:t}\big)}
       {\text{Var}\!\big(r^{(b)}_{t-W+1:t}\big)} .
  $$

---

## Intraday mechanics (live view)

### Minute-by-minute return since prior close

Let last traded price at minute t be $L_t$, and prior official **Adj Close** be $C_{\text{prev}}$.

$$r^{(\text{intra})}_t = \frac{L_t}{C_{\text{prev}}} - 1$$

In code: `intraday_return_matrix(intra_close, prev_close_map)`.

### Portfolio intraday return & drifted weights

$$r^{(p,\text{intra})}_t = \sum_i w_i \cdot r^{(\text{intra})}_{i,t}$$

$$w_i^{(\text{drift})} \propto w_i \cdot \left(1 + r^{(\text{intra})}_{i,t}\right)$$

(then renormalize to sum to 1.)

> In code: `live_portfolio_metrics(...)`.

### Synthetic “today” row for overlay

If $P^{(\text{yday})}$ is yesterday’s adjusted close and $L$ latest price:

$$
P^{(\text{today})} \;=\; P^{(\text{yday})} \cdot \frac{L}{C_{\text{prev}}}.
$$

> In code: `overlay_prices(prices_adj, last, prev_map)`.

---

## Aggregation by metadata

Group weights by a label $g(i)$ (e.g., sector or region) and normalize:

$$W_g = \sum_{i: g(i)=g} w_i$$

$$\tilde{W}_g = \frac{W_g}{\sum_{g'} W_{g'}}$$

> In code: `aggregate_by_metadata(weights, metadata, field="sector")`.

---

## Units & conventions

* Returns are **decimals** (0.05 = 5%).
* Annualization uses **252** trading days.
* VaR/ES are reported as **positive losses** (left-tail).
* Sample statistics use `ddof=1`.
* All beta/alpha/IR/TE use **aligned** dates (drop NaNs).

---

## References in code

* `metrics_core.py`: returns, NAV, annualization, Sharpe, beta/alpha, TE/IR, drawdown, covariance + shrinkage, risk contributions, optimizers, frontier, MC VaR/ES, relative series, aggregation.
* `metrics_active.py`: intraday last/close matrix, prev close map, intraday return matrix, overlay, live portfolio metrics (drifted weights + VaR nowcast).

---

*This document matches the implementation shipped with the project and is meant as a precise, beginner-friendly reference to how each number is computed.*