#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo apt-get update
sudo apt-get install -y python3 python3-pip ufw v4l-utils libgl1 libglib2.0-0t64
python3 "$root/config/generate_requirements.py" server
python3 -m pip install --user --break-system-packages -r "$root/requirements-server.txt"

port="$(python3 - "$root/python/server.env" <<'PYTHON'
import sys
from pathlib import Path

from dotenv import dotenv_values

path = Path(sys.argv[1])
defaults = """CAMERA_COUNT=2
FRONT_CAMERA=auto
BACK_CAMERA=auto
LEFT_CAMERA=auto
RIGHT_CAMERA=auto
# Use positive dimensions supported by all enabled cameras, e.g. 640x480 or 1280x720.
# Ubuntu only: set WIDTH=0 and HEIGHT=0 together to select the lowest common resolution.
WIDTH=640
HEIGHT=480
# Requested frame rate: 1-120 FPS. Choose a rate supported by your cameras, e.g. 10, 15, or 30.
FPS=10
# MJPEG (or MJPG) requests compressed capture. AUTO uses automatic format selection.
VIDEO_FORMAT=MJPEG
JPEG_QUALITY=80
SERVER_HOST=127.0.0.1
SERVER_PORT=14095
CONTROL_PORT=14098
"""
try:
    with path.open("x", encoding="utf-8") as config:
        config.write(defaults)
except FileExistsError:
    pass

try:
    settings = dotenv_values(path)
    port = int(settings.get("SERVER_PORT", 14095))
    if not 1 <= port <= 65535:
        raise ValueError("Port must be from 1 to 65535")
except (TypeError, ValueError) as exc:
    sys.exit(f"{path.name}: {exc}")
print(port)
PYTHON
)"

sudo ufw allow in proto tcp from "${NIGHT_GUARD_LAN_CIDR:-any}" to any port "$port" comment 'Night Guard server'
ufw_status="$(sudo ufw status)"
if [[ $ufw_status == 'Status: inactive'* ]]; then
    echo 'UFW is inactive. The rule is saved and will apply when UFW is enabled.'
fi
echo "Night Guard server setup complete (TCP port $port)."
