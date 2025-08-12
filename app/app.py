# app/app_streamlit.py
import datetime as dt, streamlit as st
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import TICKERS, BENCHMARKS, REFRESH_SEC_MARKET, REFRESH_SEC_CLOSED, DEFAULT_BENCH
from src.live import get_live_quotes, market_is_open

st.set_page_config(page_title="Portfolio Optimization (Live)", layout="wide")

# Initialize session state for auto-refresh
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'data' not in st.session_state:
    st.session_state.data = None

# Check market status and set refresh interval
is_open = market_is_open()
interval = REFRESH_SEC_MARKET if is_open else REFRESH_SEC_CLOSED

# Auto-refresh logic
current_time = time.time()
should_refresh = (current_time - st.session_state.last_update) >= interval

# Create placeholder for real-time updates
placeholder = st.empty()

with placeholder.container():
    st.title("Portfolio Optimization & Performance — Live")
    st.caption(f"Market open: {is_open} • Auto-refresh: {interval}s")
    
    # Show countdown timer
    time_since_update = current_time - st.session_state.last_update
    next_refresh = max(0, interval - time_since_update)
    st.caption(f"Next refresh in: {int(next_refresh)} seconds")
    
    # Fetch data if needed
    if should_refresh or st.session_state.data is None:
        try:
            with st.spinner("Fetching live data..."):
                live = get_live_quotes(TICKERS + [DEFAULT_BENCH])
                st.session_state.data = live
                st.session_state.last_update = current_time
            st.success("Live data loaded successfully!")
        except Exception as e:
            st.error(f"Could not fetch live quotes: {str(e)}")
            st.info("Using cached data instead")
    
    # Display data
    if st.session_state.data is not None:
        st.markdown(f"**Last updated:** {dt.datetime.fromtimestamp(st.session_state.last_update).strftime('%Y-%m-%d %H:%M:%S')} (server time)")
        st.dataframe(st.session_state.data, use_container_width=True)
    
    # Force refresh if it's time
    if should_refresh:
        time.sleep(1)  # Small delay to show the update
        st.rerun()