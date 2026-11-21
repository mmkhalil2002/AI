#!/usr/bin/env bash
# Configure ANY detected Wi-Fi interface (built-in or USB) on Ubuntu 24.04 using Netplan.
# Usage: sudo ./wifi-configure.sh
# Interactive: choose adapter, enter SSID and hidden password, confirm with Netplan rollback.
# Requires: bash, ip, iw, netplan, python3-yaml; run from a local console if possible.
set -Eeuo pipefail
if (( EUID != 0 )); then echo 'Run with sudo.' >&2; exit 1; fi
for c in ip iw netplan python3; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 1; }; done
python3 -c 'import yaml' || { echo 'Install python3-yaml first.' >&2; exit 1; }
[[ -t 0 ]] || { echo 'Interactive terminal required.' >&2; exit 1; }
# Loading the Pi built-in Wi-Fi driver is harmless on systems without it.
modprobe brcmfmac 2>/dev/null || true
mapfile -t IFACES < <(iw dev | awk '$1=="Interface" {print $2}')
if ((${#IFACES[@]} == 0)); then echo 'No Wi-Fi interfaces found. Check USB adapter/driver.' >&2; exit 1; fi
printf '\nAvailable Wi-Fi interfaces:\n'
for i in "${!IFACES[@]}"; do
  iface=${IFACES[$i]}
  devpath=$(readlink -f "/sys/class/net/$iface/device" 2>/dev/null || true)
  kind='onboard/other'; [[ "$devpath" == *'/usb'* ]] && kind='USB'
  printf '  %d) %s (%s)\n' "$((i+1))" "$iface" "$kind"
done
read -r -p 'Select Wi-Fi interface number: ' selection
[[ "$selection" =~ ^[0-9]+$ ]] && ((10#$selection >= 1 && 10#$selection <= ${#IFACES[@]})) || { echo 'Invalid selection.' >&2; exit 1; }
IFACE=${IFACES[$((10#$selection-1))]}
read -r -p 'Wi-Fi SSID (network name): ' SSID
[[ -n "$SSID" ]] || { echo 'SSID cannot be empty.' >&2; exit 1; }
read -r -s -p 'Wi-Fi password (blank for open network): ' PASSWORD; printf '\n'
if [[ -n "$PASSWORD" ]] && ((${#PASSWORD} < 8 || ${#PASSWORD} > 63)); then echo 'WPA passphrase must contain 8-63 characters.' >&2; exit 1; fi
printf 'Selected interface: %s\nNetwork: %s\n' "$IFACE" "$SSID"
read -r -p 'Save and try this Wi-Fi configuration? [y/N]: ' ANSWER
[[ "$ANSWER" =~ ^[Yy]$ ]] || { echo 'Canceled.'; exit 0; }
# Prefer an existing Netplan file that already defines the chosen interface;
# otherwise use a dedicated configuration file to avoid modifying Ethernet settings.
TARGET=$(IFACE="$IFACE" python3 - <<'PY'
import glob,os,yaml
iface=os.environ['IFACE']
for path in sorted(glob.glob('/etc/netplan/*.yaml')):
    try:
        with open(path) as f: cfg=yaml.safe_load(f) or {}
        if iface in cfg.get('network',{}).get('wifis',{}):
            print(path); break
    except Exception: pass
else: print('/etc/netplan/90-wifi-configure.yaml')
PY
)
EXISTED=0; [[ -f "$TARGET" ]] && EXISTED=1
BACKUP="${TARGET}.backup-$(date +%Y%m%d-%H%M%S)"
if ((EXISTED)); then cp -p "$TARGET" "$BACKUP"; echo "Backup: $BACKUP"; fi
restore() {
  if ((EXISTED)); then cp -p "$BACKUP" "$TARGET"; else rm -f "$TARGET"; fi
  netplan generate >/dev/null 2>&1 || true
}
# Password is passed to a short-lived subprocess, never echoed or saved in shell history.
if ! IFACE="$IFACE" SSID="$SSID" PASSWORD="$PASSWORD" TARGET="$TARGET" python3 - <<'PY'
import os,yaml
p=os.environ['TARGET']; iface=os.environ['IFACE']
try:
    with open(p,encoding='utf-8') as f: cfg=yaml.safe_load(f) or {}
except FileNotFoundError: cfg={}
net=cfg.setdefault('network',{}); net.setdefault('version',2)
entry=net.setdefault('wifis',{}).setdefault(iface,{})
entry['dhcp4']=True
entry['access-points']={os.environ['SSID']: ({'password':os.environ['PASSWORD']} if os.environ['PASSWORD'] else {})}
with open(p,'w',encoding='utf-8') as f: yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
PY
then restore; echo 'Could not save configuration.' >&2; exit 1; fi
chmod 600 "$TARGET"
unset PASSWORD
if ! netplan generate; then restore; echo 'Netplan validation failed; restored previous configuration.' >&2; exit 1; fi
printf '\nTrying configuration. Confirm at the Netplan prompt to keep it.\n'
printf 'If your connection drops or you do not confirm, Netplan should roll back.\n'
if ! netplan try --timeout 60; then
  restore
  echo 'Netplan try failed or was not confirmed; original file restored.' >&2
  exit 1
fi
printf '\nInterface status:\n'
ip -br addr show dev "$IFACE" || true
printf '\nWireless link:\n'
iw dev "$IFACE" link || true
if ip -4 addr show dev "$IFACE" | grep -q 'inet '; then
  echo 'SUCCESS: Wi-Fi interface has an IPv4 address.'
else
  echo 'WARNING: No IPv4 address yet. Check SSID, password, DHCP, and networkctl status.'
fi
echo "Configuration: $TARGET"
