#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo apt-get update
sudo apt-get install -y python3 python3-pip ufw v4l-utils libgl1 libglib2.0-0t64
python3 -m pip install --user --break-system-packages -c "$root/requirements.txt" opencv-python python-dotenv

port="$(python3 - "$root/python/server.env" <<'PYTHON'
import sys
from pathlib import Path

from dotenv import dotenv_values

path = Path(sys.argv[1])
defaults = """CAMERA_COUNT=4
FRONT_CAMERA=/dev/video0
BACK_CAMERA=/dev/video2
LEFT_CAMERA=/dev/video4
RIGHT_CAMERA=/dev/video6
WIDTH=0
HEIGHT=0
FPS=15
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
