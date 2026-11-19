#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/uninit-firewall-ubuntu.sh"

unit=night-guard-server.service
unit_file="/etc/systemd/system/$unit"
if [[ -e $unit_file || -L $unit_file ]]; then
    sudo systemctl disable --now "$unit"
    sudo rm -- "$unit_file"
fi
sudo systemctl daemon-reload

remove_night_guard_firewall_rules server
echo 'Night Guard server uninitialization complete.'
