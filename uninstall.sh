#!/usr/bin/env bash
# Zimo Studio - uninstall (macOS / Linux)
#
#   ./uninstall.sh          remove .venv (the only thing setup.sh created)
#   ./uninstall.sh --purge  also remove training runs, exports and caches
#
# Nothing outside this folder is touched: setup.sh never writes to your system
# Python, no services are registered, no registry entries are created.
set -euo pipefail
cd "$(dirname "$0")"

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

echo "This will remove:"
echo "  - .venv/            (the isolated environment created by setup.sh)"
[ "$PURGE" = "1" ] && echo "  - runs/  __pycache__/  .pytest_cache/  (training artifacts)"
echo

if [ -d .venv ]; then
    rm -rf .venv
    echo "[ok] removed .venv"
else
    echo "[--] .venv not present"
fi

if [ "$PURGE" = "1" ]; then
    rm -rf runs __pycache__ .pytest_cache
    find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
    echo "[ok] removed runs and caches"
else
    echo "[--] kept runs/ (pass --purge to delete training artifacts)"
fi

cat <<'EOF'

Done. The code itself is untouched; delete this folder whenever you like:
    cd .. && rm -rf "$(basename "$PWD")"
EOF
