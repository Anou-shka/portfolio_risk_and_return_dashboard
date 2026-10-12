#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The dashboard fetches daily and intraday prices on demand, so the UI is the
# only process. (src.live / src.data_fetch remain available as optional CLIs.)
exec streamlit run app/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0
