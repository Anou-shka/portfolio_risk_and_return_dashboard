# src/intraday_cache.py
import os, datetime as dt, pandas as pd
PATH="data/processed/intraday_quotes.csv"
def append_quotes(df: pd.DataFrame)->pd.DataFrame:
    d=df.copy(); d["ts_utc"]=dt.datetime.utcnow().isoformat()
    if os.path.exists(PATH): d=pd.concat([pd.read_csv(PATH), d], ignore_index=True)
    d.to_csv(PATH, index=False); return d
