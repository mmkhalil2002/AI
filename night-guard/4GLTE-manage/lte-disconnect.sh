#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-disconnect.sh
# Function: Disconnect the NightGuard-LTE data profile without disabling the modem.
# Usage: sudo ./lte-disconnect.sh
# Note: Only affects the named LTE profile; it does not disconnect Ethernet.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
need nmcli
if nmcli -t -f NAME connection show --active | grep -Fxq "$LTE_CONNECTION"; then
  nmcli connection down "$LTE_CONNECTION"
else
  echo "Cellular profile '$LTE_CONNECTION' is not active."
fi
