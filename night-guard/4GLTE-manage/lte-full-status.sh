#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-full-status.sh
# Function: Display detailed Quectel EC25 modem and SIM information.
# Usage: sudo ./lte-full-status.sh
# Requirements: ModemManager (mmcli); a modem visible to ModemManager.
# Notes: Read-only. Redacts phone number, IMEI, IMSI, ICCID, equipment ID,
#        device ID, and other identifying values from the displayed output.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  echo 'Usage: sudo ./lte-full-status.sh  (read-only detailed modem and SIM status)'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'Unexpected arguments. Use --help.' >&2; exit 2; }
id="$(modem_id)"
redact() {
  sed -E '/^[[:space:]]*\|?[[:space:]]*(device id|equipment id|imei|imsi|iccid|own|eid|msisdn|phone number|operator name):/I s@:[[:space:]]*.*@: [REDACTED]@'
}
echo '========== LTE MODEM INFORMATION =========='
mmcli -m "$id" | redact
sim="$(mmcli -m "$id" | sed -nE 's@.*primary sim path: .*/SIM/([0-9]+).*@\1@p' | head -n1)"
echo
if [[ -n "$sim" ]]; then
  echo '========== LTE SIM INFORMATION =========='
  mmcli -i "$sim" | redact
else
  echo 'No SIM information available.'
fi
