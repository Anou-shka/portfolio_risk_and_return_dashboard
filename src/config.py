# Global constants (tickers, rf rate, paths)

TICKERS = ["AAPL","MSFT","JNJ","JPM","XOM","AMZN","GOOGL","TSLA","SPY","GLD"]
BENCHMARKS = {
    "^GSPC": "S&P 500 Index (US)",
    "^STI": "Straits Times Index (Singapore)"
}

DEFAULT_BENCH = "^GSPC"  # fallback if none selected

RISK_FREE_RATE = 0.04
REFRESH_SEC_MARKET = 60  # 1 min during market hours
REFRESH_SEC_CLOSED = 900  # 15 min off-hours
