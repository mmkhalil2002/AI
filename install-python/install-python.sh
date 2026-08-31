#!/bin/bash
###############################################################################
# Build a Standalone Executable with Nuitka
# -----------------------------------------------------------------------------
# Compile the application into a single standalone executable.
# The command below:
#   • Creates a local "nuitka_cache" folder in the current directory.
#   • Automatically downloads required build tools.
#   • Does not prompt for download confirmation.
#   • Produces a standalone executable.
#
# Command:
#
# NUITKA_CACHE_DIR=$(pwd)/nuitka_cache \
# python3 -m nuitka --standalone --onefile \
# --assume-yes-for-downloads main.py
#
# OR
# python -m nuitka --standalone --onefile .\med-00-08-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766.py
###############################################################################

echo "========================================="
echo "Python Environment Installer"
echo "========================================="

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python3 is not installed."
    exit 1
fi

python3 -m ensurepip --upgrade 2>/dev/null

python3 -m pip install --upgrade pip

python3 -m pip install \
setuptools \
wheel \
virtualenv \
numpy \
scipy \
pandas \
matplotlib \
opencv-python \
pillow \
requests \
flask \
pyyaml \
psutil \
tqdm \
colorama \
pyserial \
cryptography \
scikit-learn \
joblib \
onnx \
onnxruntime \
torch \
torchvision \
torchaudio \
transformers \
accelerate \
sentencepiece \
protobuf \
huggingface_hub \
nuitka \
ordered-set \
zstandard

echo
echo "Installation Complete"

python3 -m nuitka --version