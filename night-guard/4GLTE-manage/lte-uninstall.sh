#!/usr/bin/env bash
# Script: lte-uninstall.sh
# Purpose: Reset Ubuntu LTE user-space tools for a Sixfab Quectel EC25.
# Usage: sudo ./lte-uninstall.sh
# WARNING: Temporarily interrupts cellular connectivity. Run from Ethernet/local console.
# Safety: Does NOT remove NetworkManager, kernel modules, modem firmware, or SIM settings.
set -Eeuo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo.' >&2; exit 1; }
command -v apt-get >/dev/null || { echo 'Ubuntu/Debian apt-get required.' >&2; exit 1; }
echo 'This removes modemmanager, libqmi-utils, libmbim-utils and usb-modeswitch (if installed).'
echo 'It does NOT remove NetworkManager, your Ethernet configuration, or saved connection profiles.'
echo 'WARNING: Cellular connections will be interrupted.'
read -r -p 'Type RESET-LTE to continue: ' answer
[[ $answer == RESET-LTE ]] || { echo 'Cancelled.'; exit 0; }
if systemctl list-unit-files ModemManager.service --no-legend 2>/dev/null | grep -q '^ModemManager.service'; then
  systemctl stop ModemManager.service || true
fi
packages=(modemmanager libqmi-utils libmbim-utils usb-modeswitch)
installed=()
for p in "${packages[@]}"; do
  if dpkg-query -W -f='${db:Status-Abbrev}' "$p" 2>/dev/null | grep -q '^ii'; then installed+=("$p"); fi
done
if ((${#installed[@]})); then
  apt-get purge -y "${installed[@]}"
else
  echo 'No targeted LTE packages are installed.'
fi
# Deliberately do not run apt autoremove: it could remove unrelated dependencies.
echo 'LTE package reset complete. NetworkManager and connection profiles were preserved.'
echo 'Next: sudo ./lte-install.sh'
