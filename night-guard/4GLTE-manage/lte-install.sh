#!/usr/bin/env bash
# Script: lte-install.sh
# Purpose: Install and verify Ubuntu 24.04 LTE support for Sixfab Quectel EC25.
# Usage: sudo ./lte-install.sh
# Notes: Uses Ubuntu repositories and built-in kernel drivers; does not change APN or default route.
set -Eeuo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo.' >&2; exit 1; }
command -v apt-get >/dev/null || { echo 'Ubuntu/Debian apt-get required.' >&2; exit 1; }
echo 'Installing LTE packages (NetworkManager will be installed if missing, but not reconfigured)...'
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y modemmanager network-manager libqmi-utils libmbim-utils usb-modeswitch usbutils
systemctl enable --now ModemManager.service
printf '\n=== Packages ===\n'
for p in modemmanager network-manager libqmi-utils libmbim-utils usb-modeswitch; do
  dpkg-query -W -f='${binary:Package} ${Version} ${db:Status-Abbrev}\n' "$p" 2>/dev/null || true
done
printf '\n=== Kernel modules (availability) ===\n'
for m in option qmi_wwan cdc_wdm cdc_ether; do
  if modinfo "$m" >/dev/null 2>&1; then echo "$m: available"; else echo "$m: not found in running kernel"; fi
done
printf '\n=== USB EC25 detection ===\n'
lsusb -d 2c7c:0125 || echo 'EC25 USB ID not currently detected (check USB cable/power).'
printf '\n=== ModemManager ===\n'
mmcli -L || true
printf '\n=== LTE-capable interfaces ===\n'
ip -br link
printf '\n=== USB data control devices ===\n'
find /dev -maxdepth 1 -name 'cdc-wdm*' -print 2>/dev/null || true
printf '\n=== Next steps ===\n'
echo 'If modem appears: mmcli -L; then sudo mmcli -m <ID> --enable'
echo 'Check full modem details: sudo mmcli -m <ID>'
echo 'No APN, SIM PIN, USB composition, or route was modified.'
echo 'If only ttyUSB ports appear, further USB data-mode diagnosis is required.'
