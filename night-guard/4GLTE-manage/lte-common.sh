#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-common.sh
# Function: Shared helper functions and configuration for all LTE scripts.
# This file is sourced by the other scripts; do not normally run it directly.
# Note: MODEM_ID=239 can override automatic modem discovery; LTE_CONNECTION sets the profile name.
# -----------------------------------------------------------------------------
set -euo pipefail
MODEM_ID="${MODEM_ID:-}"
LTE_CONNECTION="${LTE_CONNECTION:-NightGuard-LTE}"
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1. Install: sudo apt install modemmanager network-manager usbutils" >&2; exit 1; }; }
modem_id() {
  need mmcli
  if [[ -n "$MODEM_ID" ]]; then printf '%s\n' "$MODEM_ID"; return; fi
  local found
  found="$(mmcli -L | sed -nE 's@.*Modem/([0-9]+).*@\1@p' | head -n1)"
  [[ -n "$found" ]] || { echo 'No modem found. Check lsusb and ModemManager.' >&2; exit 1; }
  printf '%s\n' "$found"
}
