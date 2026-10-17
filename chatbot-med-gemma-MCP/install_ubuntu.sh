#!/usr/bin/env bash
# Install Python (if needed), create venv, then call config.py.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv ca-certificates
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python config.py
echo 'Installation complete.'
