#!/usr/bin/env bash
# Launch the first .py file (alphabetical order) in ../python.
# Ollama address is configured by the Python application in python/.env.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)/python"
if [[ ! -d "$PYTHON_DIR" ]]; then
  echo "[ERROR] Python directory not found: $PYTHON_DIR" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo '[ERROR] Python 3 is not installed or not on PATH.' >&2
  exit 1
fi
# Alphabetical selection matches Windows' dir /on behavior.
MAIN_PY="$(find "$PYTHON_DIR" -maxdepth 1 -type f -name '*.py' -printf '%f\n' | LC_ALL=C sort | head -n 1)"
if [[ -z "$MAIN_PY" ]]; then
  echo "[ERROR] No .py files found in $PYTHON_DIR" >&2
  exit 1
fi
cd "$PYTHON_DIR"
echo "[INFO] Running: $MAIN_PY"
echo "[INFO] Working directory: $PWD"
exec python3 "$MAIN_PY"
