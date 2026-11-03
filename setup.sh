#!/usr/bin/env bash
# Zimo Studio - one-step setup (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || {
    echo "[x] python3 not found. Install Python 3.10+ first." >&2
    exit 1
}

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/3] Creating virtual environment .venv ..."
    "$PY" -m venv .venv
else
    echo "[1/3] Reusing existing .venv"
fi

echo "[2/3] Installing dependencies (this can take a few minutes) ..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements-studio.txt

cat <<'EOF'

[3/3] Done. Start the Studio with:
    ./run.sh

Then open http://127.0.0.1:8769
EOF
