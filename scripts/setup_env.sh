#!/usr/bin/env bash
# Bash script to set up virtual environment and install dependencies

set -e

ENV_NAME="phygecnet"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.10 -y

# Detect OS to print correct activation command
echo "=== Activating conda environment ==="
# Source conda.sh to enable activation in subshell
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== Installing PyTorch with CUDA 11.8 support ==="
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing python package dependencies ==="
pip install -r requirements.txt

echo "=== Installing project package in editable mode ==="
pip install -e .

echo "=== Setup complete! ==="
echo "To activate this environment, run:"
echo "  conda activate $ENV_NAME"
