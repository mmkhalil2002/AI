#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Escape paths for systemd's ExecStart quoting and specifier expansion.
if [[ $root == *$'\n'* || $root == *$'\r'* ]]; then
    echo 'The project path must fit on one line.' >&2
    exit 1
fi
unit_root=${root//\\/\\\\}
unit_root=${unit_root//\"/\\\"}
unit_root=${unit_root//%/%%}
service_user=${SUDO_USER:-$(id -un)}
if [[ $service_user == root ]]; then
    echo 'Run this script from the Ubuntu account that installed Night Guard.' >&2
    exit 1
fi

unit=night-guard-server.service
unit_file="$(mktemp)"
trap 'rm -f -- "$unit_file"' EXIT
cat > "$unit_file" <<UNIT
[Unit]
Description=Night Guard camera server
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$service_user
SupplementaryGroups=video
ExecStart=:/usr/bin/python3 -u "${unit_root}/python/night_guard_server.py"
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
UNIT

sudo install -m 0644 "$unit_file" "/etc/systemd/system/$unit"
sudo systemctl daemon-reload
sudo systemctl enable "$unit"
sudo systemctl restart "$unit"
echo "Night Guard server is enabled at boot. View its log with journalctl -u $unit."
