#!/usr/bin/env bash
# Launch the named MedGemma patient GUI.
# Ollama address is configured by the Python application in python/.env.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)/python"
if [[ ! -d "$PYTHON_DIR" ]]; then
  echo "[ERROR] Python directory not found: $PYTHON_DIR" >&2
  exit 1
fi
MAIN_PY="$PYTHON_DIR/client-med-gemma-2w-data.py"
if [[ ! -f "$MAIN_PY" ]]; then
  echo "[ERROR] MedGemma application not found: $MAIN_PY" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys, tkinter; print(sys.executable); sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  echo '[ERROR] A working Python 3.10+ interpreter with Tkinter is unavailable.' >&2
  echo '[ACTION] Install Python 3.10 or newer and the Tkinter package (often python3-tk), then retry.' >&2
  exit 1
fi
cd "$PYTHON_DIR"
echo "[INFO] Running: $MAIN_PY"
echo "[INFO] Working directory: $PWD"
exec python3 "$MAIN_PY"
