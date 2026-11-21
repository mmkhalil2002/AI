#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-disable.sh
# Function: Disable the detected LTE modem and terminate its cellular service.
# Usage: sudo ./lte-disable.sh
# Note: Warning: any active LTE connection will be interrupted.
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lte-common.sh"
id="$(modem_id)"
echo "Disabling LTE modem $id (will terminate cellular service)..."
mmcli -m "$id" --disable
