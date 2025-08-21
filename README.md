# Portfolio Pulse — Portfolio Risk & Return Dashboard

An interactive dashboard for portfolio analysis and risk management.

It ingests **historical** and **intraday** market data, computes core stats (Sharpe, beta, volatility, drawdowns, tracking error, information ratio), builds an **efficient frontier**, and estimates **Monte Carlo VaR / ES**. The app runs continuously on **Render** with a shared `data/` folder so the UI, intraday writer, and nightly updater stay in sync.

## What You Can Do

* **Performance at a glance:** annualized return/volatility, Sharpe, beta vs benchmark, tracking error, information ratio, max drawdown
* **Cumulative NAV & relative perf:** "growth of $1" and portfolio − benchmark
* **Rolling risk:** rolling vol and rolling beta with a window slider
* **Risk decomposition:** asset-level risk contributions and total portfolio risk
* **Efficient frontier:** visualize feasible risk/return; show equal-weight, min-variance, and max-Sharpe reference points with weight bounds
* **VaR & ES (Monte Carlo):** choose confidence and horizon; see VaR and expected shortfall
* **Diversification:** sector/region pies from metadata; per-asset table with key stats
* **Live tab:** intraday portfolio vs benchmark (1-minute data during US market hours)

**Visit the live dashboard:** [Portfolio Pulse on Render](https://portfolio-risk-and-return-dashboard.onrender.com)

## Features

* Real-time portfolio risk metrics
* Interactive scenario testing
* Professional financial visualizations (Plotly)
* Sharpe Ratio, Beta, VaR/ES calculations
* Intraday cache + nightly EOD updates (yfinance → Parquet)

## Local Development

```bash
# 1) Clone and install
git clone https://github.com/Anou-shka/portfolio_risk_and_return_dashboard.git
cd portfolio_risk_and_return_dashboard
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) (Optional) Edit tickers/benchmark in src/config.py

# 3) Seed 3y history once
python -m src.data_fetch init

# 4) Start intraday writer (only during 09:30–16:00 America/New_York)
python -m src.live daemon   # or: python -m src.live tick  # one pass

# 5) Run the UI
streamlit run app/app.py

# 6) After market close, append the official daily bar
python -m src.data_fetch update
# For testing off-hours: python -m src.data_fetch update --force
```

### Environment Variables

* `PYTHONUNBUFFERED=1`
* `STREAMLIT_SERVER_HEADLESS=true`
* `APP_ACCESS_CODE=<optional password>` (enables a simple access gate in the UI)


## Access Control (Optional)

Set `APP_ACCESS_CODE` in Render. On load, the app shows a small password prompt; only users with the link **and** the code can view.
