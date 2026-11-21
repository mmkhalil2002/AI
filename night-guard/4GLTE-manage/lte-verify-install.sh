#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: lte-verify-install.sh
# Function: Verify LTE software installation, services, USB modem detection,
#           SIM presence, and whether cellular networking is available.
# Usage: ./lte-verify-install.sh
#        sudo ./lte-verify-install.sh
#        ./lte-verify-install.sh --help
# Requirements: Ubuntu 24.04; dpkg-query, systemctl, mmcli, ip, lsusb, modinfo
# Notes: Read-only. Does not enable modem, connect LTE, change routes or drivers.
#        Package installation and usable LTE internet are separate checks.
# -----------------------------------------------------------------------------
set -u

case "${1:-}" in
  -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
  '') ;;
  *) echo "Usage: $0 [--help]" >&2; exit 2 ;;
esac

passes=0; fails=0; warnings=0
pass() { printf '[PASS] %s\n' "$*"; passes=$((passes+1)); }
fail() { printf '[FAIL] %s\n' "$*"; fails=$((fails+1)); }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings+1)); }
installed() { [[ "$(dpkg-query -W -f='${db:Status-Abbrev}' "$1" 2>/dev/null || true)" == ii* ]]; }

printf '=== LTE INSTALL VERIFICATION ===\n'
printf 'Date: %s\nHost: %s\n\n' "$(date -Is)" "$(hostname)"

printf '%s\n' '-- Packages installed by lte-install.sh --'
for pkg in modemmanager network-manager libqmi-utils libmbim-utils usb-modeswitch usbutils; do
  if installed "$pkg"; then pass "$pkg is installed"; else fail "$pkg is not installed"; fi
done

printf '\n%s\n' '-- Commands and service --'
for cmd in mmcli nmcli lsusb ip; do
  if command -v "$cmd" >/dev/null 2>&1; then pass "$cmd command available"; else fail "$cmd command missing"; fi
done
if command -v systemctl >/dev/null 2>&1; then
  state=$(systemctl is-active ModemManager.service 2>/dev/null || true)
  if [[ "$state" == active ]]; then pass 'ModemManager service is active'; else fail "ModemManager service is ${state:-unknown}"; fi
  enabled=$(systemctl is-enabled ModemManager.service 2>/dev/null || true)
  if [[ "$enabled" == enabled || "$enabled" == static ]]; then pass "ModemManager service is $enabled"; else warn "ModemManager service enablement: ${enabled:-unknown}"; fi
else
  fail 'systemctl is unavailable'
fi

printf '\n%s\n' '-- Kernel driver availability --'
if command -v modinfo >/dev/null 2>&1; then
  if modinfo option >/dev/null 2>&1; then pass 'option serial driver available'; else warn 'option serial driver not found'; fi
  if modinfo qmi_wwan >/dev/null 2>&1; then pass 'qmi_wwan driver available'; else warn 'qmi_wwan driver not found (QMI mode may not be supported)'; fi
else
  warn 'modinfo unavailable; kernel modules not checked'
fi

printf '\n%s\n' '-- EC25 hardware and modem detection --'
if command -v lsusb >/dev/null 2>&1; then
  if lsusb -d 2c7c:0125 | grep -q .; then pass 'Quectel EC25 USB device detected (2c7c:0125)'; else warn 'Quectel EC25 USB device not detected'; fi
fi
if compgen -G '/dev/ttyUSB*' >/dev/null; then
  pass 'USB serial device nodes detected'
else
  warn 'No /dev/ttyUSB* nodes detected'
fi

modem_path=''
if command -v mmcli >/dev/null 2>&1; then
  modem_list=$(mmcli -L 2>&1 || true)
  modem_path=$(printf '%s\n' "$modem_list" | grep -oE '/org/freedesktop/ModemManager1/Modem/[0-9]+' | head -n 1 || true)
  if [[ -n "$modem_path" ]]; then
    pass 'ModemManager detects a modem'
    modem_id=${modem_path##*/}
    # Do not print raw mmcli output: it can contain IMEI, IMSI and phone number.
    modem_report=$(mmcli -m "$modem_id" 2>/dev/null || true)
    if printf '%s\n' "$modem_report" | grep -Eq 'model:.*EC25'; then
      pass 'Detected modem model is EC25'
    else
      warn 'First detected modem is not confirmed as EC25'
    fi
    if printf '%s\n' "$modem_report" | grep -Eq 'primary sim path: */org/freedesktop/ModemManager1/SIM/[0-9]+'; then
      pass 'Modem reports a primary SIM'
    else
      warn 'Modem does not report a primary SIM'
    fi
    state=$(printf '%s\n' "$modem_report" | sed -n 's/.*| *state: *//p' | head -n 1)
    if [[ -n "$state" ]]; then
      printf '[INFO] Modem state: %s\n' "$state"
    fi
  else
    warn 'ModemManager does not currently detect a modem'
  fi
fi

printf '\n%s\n' '-- Cellular data readiness (not required for package installation) --'
if compgen -G '/dev/cdc-wdm*' >/dev/null; then
  pass 'USB cellular control device (/dev/cdc-wdm*) exists'
else
  warn 'No /dev/cdc-wdm* device (QMI/MBIM data interface may be missing; ECM may use a different interface)'
fi
if command -v ip >/dev/null 2>&1; then
  if ip -o link show | grep -Eq ':[[:space:]]*(wwan[0-9]+|usb[0-9]+):'; then
    pass 'Candidate cellular network interface detected'
  else
    warn 'No wwan*/usb* network interface found (other names are possible)'
  fi
  if ip link show eth0 >/dev/null 2>&1; then
    if [[ "$(cat /sys/class/net/eth0/operstate 2>/dev/null || true)" == up ]]; then pass 'eth0 Ethernet link remains UP'; else warn 'eth0 Ethernet link is not UP'; fi
  fi
fi

printf '\n=== SUMMARY ===\n'
printf 'PASS: %d | FAIL: %d | WARN: %d\n' "$passes" "$fails" "$warnings"
if ((fails > 0)); then
  printf 'RESULT: INSTALL VERIFICATION FAILED for required software checks.\n'
  exit 1
fi
printf 'RESULT: LTE management software installation verified for checked components.\n'
printf 'NOTE: Warnings may indicate incomplete modem/data readiness; LTE internet is NOT verified by this script.\n'
exit 0
