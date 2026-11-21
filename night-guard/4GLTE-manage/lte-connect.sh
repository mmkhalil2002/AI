#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-connect.sh
# Function: Enable the modem and create/update/activate an LTE data profile.
# Usage: sudo ./lte-connect.sh [APN]  (default: fast.t-mobile.com)
# Note: Requires ModemManager, NetworkManager and a working cellular data interface.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
need nmcli
id="$(modem_id)"
APN="${1:-${LTE_APN:-fast.t-mobile.com}}"
[[ "$APN" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid APN'; exit 2; }
echo "Enabling modem $id..."
mmcli -m "$id" --enable
if ! nmcli -t -f NAME connection show | grep -Fxq "$LTE_CONNECTION"; then
  echo "Creating cellular profile '$LTE_CONNECTION' with APN '$APN'..."
  nmcli connection add type gsm ifname '*' con-name "$LTE_CONNECTION" apn "$APN" connection.autoconnect no ipv4.route-metric 700 ipv6.route-metric 700
else
  echo "Updating APN for existing profile '$LTE_CONNECTION'..."
  nmcli connection modify "$LTE_CONNECTION" gsm.apn "$APN" connection.autoconnect no ipv4.route-metric 700 ipv6.route-metric 700
fi
echo 'Attempting cellular connection...'
if ! nmcli --wait 45 connection up "$LTE_CONNECTION"; then
  echo 'Connection failed. If mmcli lists only ttyUSB ports and no wwan/usb data interface, check USB networking drivers/mode with lte-diagnostics.sh.' >&2
  exit 1
fi
echo 'Connected. Cellular interface:'
nmcli -f GENERAL.DEVICES,GENERAL.STATE,IP4.ADDRESS,IP4.GATEWAY connection show "$LTE_CONNECTION"
