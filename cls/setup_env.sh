#!/usr/bin/env bash
# Create the project virtual environment and check the GPUs.
#
# Usage (from the tea_kd folder):
#   bash setup_env.sh                 # creates ./.venv
#   VENV_DIR=/path/to/venv bash setup_env.sh
#   PYTHON=python3.11 bash setup_env.sh
#
# Afterwards, in every new terminal / tmux window:
#   source .venv/bin/activate
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"   # matches driver 550 / CUDA 12.4
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"

# --- 1. Python version check (PyTorch 2.5.1 wheels: Python 3.9 - 3.12)
PYV=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYV" in
  3.9|3.10|3.11|3.12) echo "Python $PYV found ($PYTHON)";;
  *) echo "ERROR: Python $PYV is not supported by torch $TORCH_VERSION. Use 3.10-3.12, e.g. PYTHON=python3.11 bash setup_env.sh"; exit 1;;
esac

# --- 2. Create the venv
if [ ! -d "$VENV_DIR" ]; then
  if ! "$PYTHON" -m venv "$VENV_DIR"; then
    echo "ERROR: could not create a venv. On Ubuntu ask an admin for: sudo apt install python${PYV}-venv"
    echo "       (or install 'uv' without sudo and run: uv venv $VENV_DIR --python $PYV)"
    exit 1
  fi
  echo "Created $VENV_DIR"
else
  echo "Reusing existing $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools -q

# --- 3. PyTorch (CUDA 12.4 build) + the rest
pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
pip install -r requirements.txt

# --- 4. Check that PyTorch sees the GPUs with nvidia-smi numbering
python - <<'EOF'
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import torch, torchvision
print(f"\ntorch {torch.__version__}, torchvision {torchvision.__version__}, CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch cannot see a GPU. Check the driver and that the cu124 wheel was installed.")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i):<24} {free/2**30:5.1f} / {total/2**30:4.1f} GB free")
x = torch.randn(1024, 1024, device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = x @ x
print("bf16 matmul OK on", y.device)
EOF

echo -e "\nDone. Activate with:  source $VENV_DIR/bin/activate"
