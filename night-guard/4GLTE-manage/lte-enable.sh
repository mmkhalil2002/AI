#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-enable.sh
# Function: Enable the detected Quectel EC25 modem using ModemManager.
# Usage: sudo ./lte-enable.sh
# Note: Enabling does not automatically establish a cellular data connection.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
id="$(modem_id)"
echo "Enabling LTE modem $id..."
mmcli -m "$id" --enable
mmcli -m "$id" | grep -Ei 'state:|power state:|access tech:|signal quality:|registration:' || true
