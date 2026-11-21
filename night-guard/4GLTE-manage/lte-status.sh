#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-status.sh
# Function: Read-only report of LTE modem, SIM, network interfaces and connection status.
# Usage: sudo ./lte-status.sh
# Note: Does not enable or connect the modem.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
id="$(modem_id)"
echo "=== LTE modem (ID $id) ==="
mmcli -m "$id" | sed -E '/(device id|equipment id|imei|imsi|iccid|own):/I s@:[[:space:]]*.*@: [REDACTED]@'
echo; echo '=== SIM ==='
sim="$(mmcli -m "$id" | sed -nE 's@.*primary sim path: .*/SIM/([0-9]+).*@\1@p' | head -n1)"
if [[ -n "$sim" ]]; then mmcli -i "$sim" | sed -E '/imsi:|iccid:/d'; else echo 'SIM not reported'; fi
echo; echo '=== Network interfaces ==='
ip -br link
echo; echo '=== NetworkManager devices ==='
if command -v nmcli >/dev/null; then nmcli device status; else echo 'nmcli not installed'; fi
echo; echo '=== LTE connection ==='
if command -v nmcli >/dev/null; then nmcli -f GENERAL.STATE,GENERAL.CONNECTION device show 2>/dev/null | head -n 35 || true; fi
