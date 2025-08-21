#!/usr/bin/env bash
set -e

# Shared data dirs
mkdir -p data/raw/intraday data/processed

# Seed 3y history once (idempotent)
python -m src.data_fetch init || true

# Intraday daemon (background)
python -m src.live daemon --interval 60 &

# Nightly EOD updater (background)
python -m src.auto_updater &

# Streamlit UI (foreground keeps service alive)
streamlit run app/app.py --server.port "$PORT" --server.address 0.0.0.0
