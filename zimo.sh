#!/usr/bin/env bash
# 启动 Zimo Studio。软链到 PATH 里之后，任意目录敲 zimo 即可：
#   ln -sf "$PWD/zimo.sh" /usr/local/bin/zimo
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python zimo_run.py "$@"
else
    exec python3 zimo_run.py "$@"
fi
