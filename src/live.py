# src/live.py
import datetime as dt, pandas as pd, yfinance as yf, streamlit as st

@st.cache_data(ttl=30)
def get_live_quotes(tickers: list[str]) -> pd.DataFrame:
    rows=[]
    for t in tickers:
        try:
            y=yf.Ticker(t)
            fi=getattr(y,"fast_info",None)
            price=float(fi.last_price) if fi and fi.last_price is not None else float(y.history(period="1d", interval="1m")["Close"].iloc[-1])
        except Exception:
            price=float("nan")
        currency = getattr(y, "currency", "USD")
        rows.append({"ticker": t, "Last Price": price, "Currency": currency})
    return pd.DataFrame(rows)

def market_is_open() -> bool:
    now_utc=dt.datetime.utcnow()
    et=now_utc - dt.timedelta(hours=4)  # simple DST-ish heuristic
    return (et.weekday()<5) and ((et.hour>9 or (et.hour==9 and et.minute>=30)) and et.hour<16)
