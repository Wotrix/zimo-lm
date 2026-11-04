#!/usr/bin/env bash
# Zimo Studio - one-step setup (macOS / Linux / WSL)
#
#   ./setup.sh              # auto: CUDA build when an NVIDIA GPU is present
#   ./setup.sh cpu          # force the small CPU-only wheel
#   ./setup.sh cu124        # force a specific CUDA build (cu121/cu124/cu126)
#   ./setup.sh --check      # print what would be installed, change nothing
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-auto}"
CHECK_ONLY=0
if [ "$MODE" = "--check" ]; then
    CHECK_ONLY=1
    MODE=auto
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
    elif command -v python >/dev/null 2>&1; then
        PY=python   # Windows / Git Bash
    else
        echo "[x] python3 not found. Install Python 3.10+ first." >&2
        exit 1
    fi
fi

# ---- detect the accelerator and pick a torch index -------------------------
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
fi

case "$MODE" in
    cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    cu12*) TORCH_INDEX="https://download.pytorch.org/whl/${MODE}" ;;
    auto)
        if [ -n "$GPU_NAME" ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        fi
        ;;
    *) echo "[x] unknown mode: $MODE (use auto|cpu|cu124)" >&2; exit 1 ;;
esac
TORCH_INDEX="${ZIMO_TORCH_INDEX:-$TORCH_INDEX}"

echo "Python      : $($PY -V 2>&1)"
echo "GPU         : ${GPU_NAME:-none detected}"
echo "torch index : $TORCH_INDEX"
if [ -n "$GPU_NAME" ] && [ "$TORCH_INDEX" = "https://download.pytorch.org/whl/cpu" ]; then
    echo "              (GPU present but CPU wheel forced)"
fi
if [ -z "$GPU_NAME" ] && [[ "$TORCH_INDEX" == *cu1* ]]; then
    echo "              (no NVIDIA GPU found - CUDA build will fall back to CPU at runtime)"
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo "(check only, nothing installed)"
    exit 0
fi

# ---- venv -----------------------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
    echo "[1/3] Creating virtual environment .venv ..."
    "$PY" -m venv .venv
else
    echo "[1/3] Reusing existing .venv"
fi

echo "[2/3] Installing PyTorch from the index above (CUDA wheels are ~2.5 GB) ..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install torch --index-url "$TORCH_INDEX"

echo "[3/3] Installing the Studio dependencies ..."
.venv/bin/python -m pip install -r requirements-studio.txt

.venv/bin/python - <<'PY'
import torch
print()
print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda or "cpu-only")
print("available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
PY

cat <<'EOF'

Done. Start the Studio with:
    ./run.sh

Then open http://127.0.0.1:8769
EOF
