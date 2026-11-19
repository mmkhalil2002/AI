#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/uninit-firewall-ubuntu.sh"

# Older launchers installed a service in the desktop user's home directory.
unit=night-guard-client.service
unit_file="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$unit"
if [[ -e $unit_file || -L $unit_file ]]; then
    systemctl --user disable --now "$unit"
    rm -- "$unit_file"
    systemctl --user daemon-reload
fi

remove_night_guard_firewall_rules client
echo 'Night Guard client uninitialization complete.'
