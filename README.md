# Portfolio Pulse — Portfolio Risk & Return Dashboard

An interactive dashboard for portfolio analysis and risk management.

It pulls **historical** (3-year daily) and **intraday** (1-minute) market data from Yahoo Finance on demand, computes core stats (Sharpe, beta, volatility, drawdowns, tracking error, information ratio), builds an **efficient frontier**, and estimates **Monte Carlo VaR / ES**. The Streamlit UI is the only process — no background daemons or persistent disk are needed, so it runs on Render's free plan.

## What You Can Do

* **Performance at a glance:** annualized return/volatility, Sharpe, beta vs benchmark, tracking error, information ratio, max drawdown
* **Cumulative NAV & relative perf:** "growth of $1" and portfolio − benchmark
* **Rolling risk:** rolling vol and rolling beta with a window slider
* **Risk decomposition:** asset-level risk contributions and total portfolio risk
* **Efficient frontier:** visualize feasible risk/return; show equal-weight, min-variance, and max-Sharpe reference points with weight bounds
* **VaR & ES (Monte Carlo):** choose confidence and horizon; see VaR and expected shortfall
* **Diversification:** sector/region pies from metadata; per-asset table with key stats
* **Live tab:** intraday portfolio vs benchmark (1-minute data). Each asset is measured against the close before its own latest session, so mixed stock + crypto portfolios work on weekends and outside US hours

**Visit the live dashboard:** [Portfolio Pulse on Render](https://portfolio-pulse.onrender.com)

## Features

* Real-time portfolio risk metrics
* Interactive scenario testing
* Professional financial visualizations (Plotly)
* Sharpe Ratio, Beta, VaR/ES calculations
* Stocks and crypto analysed on one trading calendar (crypto sampled on stock trading days) so 252-day annualisation is valid

## Local Development

```bash
# 1) Clone and install
git clone https://github.com/Anou-shka/portfolio_risk_and_return_dashboard.git
cd portfolio_risk_and_return_dashboard
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run the UI (same command Render uses)
bash scripts/start.sh            # or: streamlit run app/app.py
```

> Never commit `.venv/` — Render builds its own virtualenv at that exact path, so a committed one breaks deploys.

### Optional offline data tools

The app does not need these; they snapshot data to Parquet for offline analysis.

```bash
python -m src.data_fetch init     # 3y daily history → data/processed/
python -m src.data_fetch update   # append the official daily bar after the close (--force off-hours)
python -m src.live daemon         # 1-minute cache → data/raw/intraday/ (US market hours)
```

## Deploying on Render

`render.yaml` is a ready Blueprint (free plan, Python 3.12, start command `bash scripts/start.sh`).
If the service was created manually, set **Start Command** to `bash scripts/start.sh` and
**Build Command** to `pip install -r requirements.txt` in the service settings.

### Environment Variables

* `PYTHONUNBUFFERED=1`
* `STREAMLIT_SERVER_HEADLESS=true`
* `APP_ACCESS_CODE=<optional password>` (enables a simple access gate in the UI)
* `PP_VALIDATE_METRICS=1` (optional: log a SPY-vs-QQQ metrics self-check once per session)


## Access Control (Optional)

Set `APP_ACCESS_CODE` in Render. On load, the app shows a small password prompt; only users with the link **and** the code can view.
