#!/usr/bin/env bash
# Zimo Studio - launch (macOS / Linux). Run ./setup.sh once first.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[x] .venv not found. Run ./setup.sh first." >&2
    exit 1
fi

echo "Starting Zimo Studio on http://127.0.0.1:8769 ..."
exec .venv/bin/python -m streamlit run app.py \
    --server.port 8769 --server.address 127.0.0.1 \
    --browser.gatherUsageStats false
