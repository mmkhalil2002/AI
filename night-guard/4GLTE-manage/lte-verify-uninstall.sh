#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-verify-uninstall.sh
# Purpose: Verify that LTE management software was removed after running
#          lte-uninstall.sh on Ubuntu 24.04 / Raspberry Pi 5 / Quectel EC25.
# Usage:  ./lte-verify-uninstall.sh
#         sudo ./lte-verify-uninstall.sh
# Behavior: Read-only. Does not remove packages, stop services, or change links.
# Notes:   NetworkManager and built-in kernel USB drivers should remain installed.
#          A detected EC25 USB device or /dev/ttyUSB* is NOT an uninstall failure.
# -----------------------------------------------------------------------------
set -u

case "${1:-}" in
  -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
  '') ;;
  *) echo "Usage: $0 [--help]" >&2; exit 2 ;;
esac

passes=0
fails=0
warnings=0
pass() { printf '[PASS] %s\n' "$*"; passes=$((passes+1)); }
fail() { printf '[FAIL] %s\n' "$*"; fails=$((fails+1)); }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings+1)); }

package_status() {
    local pkg="$1" status
    status=$(dpkg-query -W -f='${db:Status-Abbrev}' "$pkg" 2>/dev/null || true)
    case "$status" in
        ii*) fail "$pkg is still installed ($status)" ;;
        rc*) warn "$pkg removed but residual configuration remains ($status)" ;;
        '')  pass "$pkg is not installed" ;;
        *)   pass "$pkg is not fully installed ($status)" ;;
    esac
}

printf '=== LTE UNINSTALL VERIFICATION ===\n'
printf 'Date: %s\n' "$(date -Is)"
printf 'Host: %s\n\n' "$(hostname)"

printf '%s\n' '-- LTE software packages --'
package_status modemmanager
package_status libqmi-utils
package_status libmbim-utils
package_status usb-modeswitch
# NetworkManager is intentionally excluded: it is a system network manager.

printf '\n%s\n' '-- ModemManager executable and service --'
if command -v mmcli >/dev/null 2>&1; then
    fail "mmcli is still available at $(command -v mmcli)"
else
    pass 'mmcli command is absent'
fi

if command -v systemctl >/dev/null 2>&1; then
    state=$(systemctl is-active ModemManager.service 2>/dev/null || true)
    if [[ "$state" == active ]]; then
        fail 'ModemManager service is active'
    else
        pass "ModemManager service is not active (${state:-unknown})"
    fi
else
    warn 'systemctl unavailable; service state not checked'
fi

printf '\n%s\n' '-- Network safety checks --'
if command -v ip >/dev/null 2>&1; then
    if ip link show eth0 >/dev/null 2>&1; then
        if [[ "$(cat /sys/class/net/eth0/operstate 2>/dev/null || true)" == up ]]; then
            pass 'eth0 Ethernet link is UP'
        else
            warn 'eth0 exists but its Ethernet link is not UP'
        fi
    else
        warn 'eth0 not found (may use another interface name)'
    fi
else
    warn 'ip command unavailable; Ethernet check skipped'
fi

if dpkg-query -W -f='${db:Status-Abbrev}' network-manager 2>/dev/null | grep -q '^ii'; then
    pass 'NetworkManager remains installed (intentionally preserved)'
else
    warn 'NetworkManager is not installed; this may be normal for Ubuntu Server'
fi

printf '\n%s\n' '-- USB modem detection (informational only) --'
if command -v lsusb >/dev/null 2>&1; then
    if lsusb -d 2c7c:0125 | grep -q .; then
        printf '[INFO] Quectel EC25 is still visible on USB; kernel detection is expected.\n'
    else
        printf '[INFO] Quectel EC25 not currently visible on USB.\n'
    fi
else
    printf '[INFO] lsusb unavailable; USB detection not checked.\n'
fi
if compgen -G '/dev/ttyUSB*' >/dev/null; then
    printf '[INFO] USB serial ports exist; kernel drivers were intentionally retained.\n'
fi

printf '\n=== SUMMARY ===\n'
printf 'PASS: %d | FAIL: %d | WARN: %d\n' "$passes" "$fails" "$warnings"
if (( fails > 0 )); then
    printf 'RESULT: UNINSTALL VERIFICATION FAILED. Review FAIL lines above.\n'
    exit 1
fi
printf 'RESULT: LTE management software removal verified for checked components.\n'
printf 'NOTE: This does not prove every possible LTE configuration was removed.\n'
exit 0
