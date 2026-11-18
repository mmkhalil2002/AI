#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo apt-get update
sudo apt-get install -y python3 python3-pip ufw libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libdbus-1-3 libfontconfig1
python3 -m pip install --user --break-system-packages -c "$root/requirements.txt" requests PySide6 python-dotenv

port="$(python3 - "$root/python/client.env" <<'PYTHON'
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

path = Path(sys.argv[1])
defaults = """SERVER_URL=http://127.0.0.1:14095
GUI_TITLE=Night Guard
GUI_WIDTH=880
GUI_HEIGHT=620
WIDTH=480
HEIGHT=320
CONTROL_PANEL_WIDTH=200
"""
try:
    with path.open("x", encoding="utf-8") as config:
        config.write(defaults)
except FileExistsError:
    pass

try:
    settings = dotenv_values(path)
    address = (settings.get("SERVER_URL") or "").strip() or "http://127.0.0.1:14095"
    explicit_scheme = "://" in address
    parsed = urlsplit(address if explicit_scheme else "http://" + address)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid SERVER_URL")
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}[parsed.scheme] if explicit_scheme else 14095
    if not 1 <= port <= 65535:
        raise ValueError("Port must be from 1 to 65535")
except (TypeError, ValueError) as exc:
    sys.exit(f"{path.name}: {exc}")
print(port)
PYTHON
)"

sudo ufw allow out proto tcp to any port "$port" comment 'Night Guard client'
ufw_status="$(sudo ufw status)"
if [[ $ufw_status == 'Status: inactive'* ]]; then
    echo 'UFW is inactive. The rule is saved and will apply when UFW is enabled.'
fi
echo "Night Guard client setup complete (TCP port $port)."
