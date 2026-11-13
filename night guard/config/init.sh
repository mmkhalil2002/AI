#!/usr/bin/env bash
# Night Guard Ubuntu 24.04 setup; run: bash config/init.sh
# Does not start the video server or GUI. Does not create a virtualenv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$ROOT/requirements.txt"
PYDIR="$ROOT/python"
echo "[INFO] Night Guard Ubuntu setup: $ROOT"
[[ -f "$REQ" ]] || { echo "[ERROR] Missing $REQ" >&2; exit 1; }
mkdir -p "$PYDIR"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-opencv python3-numpy v4l-utils python3-venv
# Ubuntu 24.04 protects OS-managed Python. Use pip --break-system-packages only
# when needed; no venv is created. Prefer apt OpenCV to avoid conflicts.
python3 -m pip install --user --break-system-packages -r "$REQ"
if [[ ! -f "$PYDIR/server.env" ]]; then
  cat > "$PYDIR/server.env" <<'ENV'
CAMERA_COUNT=4
FRONT_CAMERA=/dev/video0
BACK_CAMERA=/dev/video2
LEFT_CAMERA=/dev/video4
RIGHT_CAMERA=/dev/video6
WIDTH=640
HEIGHT=480
FPS=15
JPEG_QUALITY=80
SERVER_HOST=127.0.0.1
SERVER_PORT=14095
CONTROL_PORT=14098
ENV
  echo '[INFO] Created python/server.env; edit camera paths before use.'
fi
if [[ ! -f "$PYDIR/client.env" ]]; then
  cat > "$PYDIR/client.env" <<'ENV'
SERVER_URL=http://127.0.0.1:14095
GUI_TITLE=Night Guard
GUI_WIDTH=1400
GUI_HEIGHT=900
CONTROL_PANEL_HEIGHT=200
ENV
  echo '[INFO] Created python/client.env; set server LAN IP on Windows client.'
fi
python3 -c 'import cv2; print("OpenCV:", cv2.__version__)'
echo '[INFO] USB video devices:'
v4l2-ctl --list-devices || true
echo '[OK] Setup complete. No application started.'
echo 'Server: python3 python/night_guard_server.py -v 2 -p 14095 -c 14098'
