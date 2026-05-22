#!/usr/bin/env bash
# Create a CUDA-enabled venv for gebru (8x Quadro RTX 6000).
#
# Usage (run from project root):
#   bash cuda/setup.sh
#
# What it does:
#   1. Detects the installed CUDA toolkit version via nvcc.
#   2. Installs PyTorch with the matching CUDA wheel.
#   3. Installs all other deps from cuda/requirements.txt.
#
# The resulting venv is at cuda/.venv and is used by all cuda/ run scripts.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="cuda/.venv"

if [ -d "$VENV" ]; then
    echo "venv already exists at $VENV — remove it first if you want a clean install."
    exit 1
fi

# Detect Python
PY="${PY:-python3}"
echo "Using Python: $($PY --version)"

# Detect CUDA toolkit version
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_MAJ="${CUDA_VER%%.*}"
    CUDA_MIN="${CUDA_VER##*.}"
    echo "Detected CUDA toolkit: $CUDA_VER  (major=$CUDA_MAJ minor=$CUDA_MIN)"
else
    echo "nvcc not found on PATH. Falling back to CUDA 12.6 wheel."
    CUDA_MAJ=12; CUDA_MIN=6
fi

# Map toolkit version to the closest available PyTorch wheel suffix.
# Wheels exist for: cu118, cu121, cu124, cu126, cu128, (cu130+ TBD).
if   [ "$CUDA_MAJ" -ge 13 ]; then
    CU="cu128"   # use latest known wheel; update when cu130+ wheels appear
elif [ "$CUDA_MAJ" -eq 12 ] && [ "$CUDA_MIN" -ge 8 ]; then
    CU="cu128"
elif [ "$CUDA_MAJ" -eq 12 ] && [ "$CUDA_MIN" -ge 6 ]; then
    CU="cu126"
elif [ "$CUDA_MAJ" -eq 12 ] && [ "$CUDA_MIN" -ge 4 ]; then
    CU="cu124"
elif [ "$CUDA_MAJ" -eq 12 ] && [ "$CUDA_MIN" -ge 1 ]; then
    CU="cu121"
else
    CU="cu118"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CU}"
echo "PyTorch wheel suffix: +${CU}  index: $TORCH_INDEX"

# Create venv
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel

# Install PyTorch with CUDA support
"$VENV/bin/pip" install torch --extra-index-url "$TORCH_INDEX"

# Install remaining deps (gymnasium, minigrid, tensorboard, etc.)
# Exclude any torch line from cuda/requirements.txt since we installed it above
grep -v '^torch' cuda/requirements.txt | grep -v '^#' | grep -v '^$' \
    | "$VENV/bin/pip" install -r /dev/stdin

echo
echo "Setup complete. Verify with:"
echo "  $VENV/bin/python -c \"import torch; print(torch.version.cuda, torch.cuda.get_device_name(0))\""
