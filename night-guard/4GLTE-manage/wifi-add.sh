#!/usr/bin/env bash
# Configure ANY detected Wi-Fi interface (built-in or USB) on Ubuntu 24.04 using Netplan.
# Usage: sudo ./wifi-add.sh
# Interactive: choose adapter, enter SSID and hidden password, confirm with Netplan rollback.
# Requires: bash, ip, iw, netplan, python3-yaml; run from a local console if possible.
set -Eeuo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: sudo ./wifi-add.sh  # select any built-in or USB Wi-Fi interface, persist Raspberry Pi onboard Wi-Fi driver, enter SSID and password, repair WPA service conflicts, and test connection'
  exit 0
fi
if (($#)); then echo 'Unknown option; use --help' >&2; exit 2; fi
if (( EUID != 0 )); then echo 'Run with sudo.' >&2; exit 1; fi
for c in ip iw netplan python3; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 1; }; done
python3 -c 'import yaml' || { echo 'Install python3-yaml first.' >&2; exit 1; }
[[ -t 0 ]] || { echo 'Interactive terminal required.' >&2; exit 1; }
# Raspberry Pi onboard Wi-Fi: ensure the Broadcom driver loads at every boot.
# Only write the persistent modules-load configuration on Raspberry Pi hardware.
# This does not modify USB Wi-Fi naming, Netplan, Ethernet, or LTE.
configure_pi_onboard_wifi() {
  local model module_file='/etc/modules-load.d/brcmfmac.conf'
  model=$(tr -d '\000' </proc/device-tree/model 2>/dev/null || true)
  if [[ "$model" != *'Raspberry Pi'* ]]; then
    echo 'Non-Raspberry Pi hardware (or model unavailable); skipping persistent Broadcom driver configuration.'
    return 0
  fi
  echo "=== Raspberry Pi onboard Wi-Fi boot check ($model) ==="
  if ! command -v modprobe >/dev/null 2>&1; then
    echo 'WARNING: modprobe unavailable; cannot initialize onboard Wi-Fi.' >&2
    return 0
  fi
  if ! modprobe brcmfmac; then
    echo 'WARNING: brcmfmac could not load; inspect firmware and kernel logs.' >&2
  fi
  if [[ ! -f "$module_file" ]] || ! grep -qxF 'brcmfmac' "$module_file"; then
    if [[ -e "$module_file" ]]; then
      echo "WARNING: $module_file already exists with other contents; leaving unchanged." >&2
      echo '         Add brcmfmac to this file manually after reviewing it.' >&2
    elif printf 'brcmfmac\n' > "$module_file"; then
      chmod 644 "$module_file"
      echo "Installed boot-time driver load: $module_file"
    else
      echo "WARNING: Cannot write $module_file" >&2
    fi
  else
    echo "Boot-time driver loading already configured: $module_file"
  fi
  if [[ -d /sys/class/net/wlan0 ]]; then
    echo 'PASS: built-in wlan0 exists.'
    ip -4 -br addr show wlan0 || true
  else
    echo 'WARNING: wlan0 is missing even after loading brcmfmac.' >&2
    echo 'Recent Broadcom driver messages:'
    dmesg 2>/dev/null | grep -i 'brcmfmac' | tail -n 12 || true
  fi
  echo 'Boot loader status:'
  systemctl is-enabled systemd-modules-load.service 2>/dev/null || true
  echo 'The systemd-modules-load service is normally static and starts automatically; no enable command is needed.'
}
configure_pi_onboard_wifi
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

# Repair only the selected interface's Netplan WPA service. A socket file can
# remain after an unclean shutdown; a live socket must NEVER be unlinked.
SERVICE="netplan-wpa-${IFACE}.service"
SOCKET="/run/wpa_supplicant/${IFACE}"
log_wpa_owner() {
  echo "Processes mentioning selected interface ($IFACE):" >&2
  ps -eo pid=,args= | awk -v iface="$IFACE" '$0 ~ /[w]pa_supplicant/ && index($0,iface) {print}' >&2 || true
  if command -v ss >/dev/null; then
    ss -xlpne 2>/dev/null | grep -F -- "$SOCKET" >&2 || true
  fi
}
# A process can be bound to a UNIX socket even if fuser doesn't report it.
# Conversely, a stale pathname can exist without a listening process.
socket_live() {
  [[ -S "$SOCKET" ]] || return 1
  if command -v ss >/dev/null && ss -xlpne 2>/dev/null | grep -Fq -- "$SOCKET"; then return 0; fi
  if command -v fuser >/dev/null && fuser "$SOCKET" >/dev/null 2>&1; then return 0; fi
  if command -v wpa_cli >/dev/null && wpa_cli -p /run/wpa_supplicant -i "$IFACE" ping 2>/dev/null | grep -qx PONG; then return 0; fi
  # If we cannot check owners reliably, do not risk unlinking a live socket.
  if ! command -v ss >/dev/null && ! command -v fuser >/dev/null && ! command -v wpa_cli >/dev/null; then return 0; fi
  return 1
}
# A global wpa_supplicant (-u) can legitimately own the selected adapter's
# socket. Ask it to release ONLY that interface through its D-Bus API (or its
# global control socket); do not kill the daemon or unlink a live socket.
release_global_wpa_interface() {
  local owner_pids pid cmd output path
  [[ -S "$SOCKET" ]] || return 1
  command -v fuser >/dev/null || { echo 'Cannot identify socket owner (fuser missing).' >&2; return 1; }
  owner_pids=$(fuser "$SOCKET" 2>/dev/null || true)
  [[ -n "$owner_pids" ]] || return 1
  for pid in $owner_pids; do
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    if [[ "$cmd" != *wpa_supplicant* || "$cmd" != *' -u '* ]]; then
      echo "Socket owned by non-global WPA process $pid; leaving it untouched." >&2
      return 1
    fi
  done
  # Do not take over a Wi-Fi interface actively controlled by NetworkManager.
  if command -v nmcli >/dev/null; then
    local managed state
    managed=$(nmcli -g GENERAL.NM-MANAGED device show "$IFACE" 2>/dev/null || true)
    state=$(nmcli -g GENERAL.STATE device show "$IFACE" 2>/dev/null || true)
    if [[ "$managed" == yes && "$state" != 30* && "$state" != 20* && "$state" != 10* ]]; then
      echo 'NetworkManager appears to control this interface; refusing takeover.' >&2
      return 1
    fi
  fi
  echo "Global WPA daemon owns $IFACE; requesting interface-specific release..."
  # Preferred: D-Bus API of the global wpa_supplicant. The object path comes
  # directly from GetInterface, not from arbitrary user input.
  if command -v busctl >/dev/null; then
    output=$(busctl --system call fi.w1.wpa_supplicant1 /fi/w1/wpa_supplicant1 fi.w1.wpa_supplicant1 GetInterface s "$IFACE" 2>/dev/null || true)
    if [[ "$output" =~ ^o[[:space:]]+\"([^\"]+)\" ]]; then
      path=${BASH_REMATCH[1]}
      if [[ "$path" == /fi/w1/wpa_supplicant1/Interfaces/* ]] && busctl --system call fi.w1.wpa_supplicant1 /fi/w1/wpa_supplicant1 fi.w1.wpa_supplicant1 RemoveInterface o "$path" >/dev/null 2>&1; then
        echo 'Released selected interface using WPA D-Bus API.'
      else
        echo 'WPA D-Bus release unsuccessful.' >&2
      fi
    fi
  fi
  # Alternative for global WPA instances that expose a global control socket.
  if socket_live && command -v wpa_cli >/dev/null && [[ -S /run/wpa_supplicant/global ]]; then
    if [[ $(wpa_cli -g /run/wpa_supplicant/global interface_remove "$IFACE" 2>/dev/null | tail -n 1) == OK ]]; then
      echo 'Released selected interface through global WPA control socket.'
    fi
  fi
  for _ in 1 2 3 4 5; do
    socket_live || return 0
    sleep 1
  done
  echo 'Global WPA daemon did not release the interface; preserving its socket.' >&2
  return 1
}
repair_wpa_socket() {
  echo "Checking WPA service for $IFACE..."
  # Stop only the selected adapter's service and allow it time to release socket.
  systemctl stop "$SERVICE" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! socket_live; then break; fi
    sleep 1
  done
  if [[ -S "$SOCKET" ]]; then
    if socket_live && ! release_global_wpa_interface; then
      echo "WARNING: A live process still owns $SOCKET." >&2
      log_wpa_owner
      echo 'Leaving the live socket untouched. Stop the competing manager for this adapter.' >&2
      return 1
    fi
    # The global process may have removed its interface and socket already.
    if [[ ! -S "$SOCKET" ]]; then
      systemctl reset-failed "$SERVICE" 2>/dev/null || true
      return 0
    fi
    # A process mentioning the interface may use another control socket; avoid
    # deleting a socket if an untracked supplicant might still own it.
    if ps -eo args= | awk -v iface="$IFACE" '$0 ~ /[w]pa_supplicant/ && index($0,iface) {found=1} END {exit !found}'; then
      echo "WARNING: An independent WPA process still references $IFACE." >&2
      log_wpa_owner
      return 1
    fi
    echo "Removing confirmed stale WPA socket: $SOCKET"
    rm -f -- "$SOCKET"
  fi
  systemctl reset-failed "$SERVICE" 2>/dev/null || true
}
# Do not preemptively stop the selected adapter if an unrelated manager owns it.
# netplan try will start/restart its service; repair only if that service fails.
printf '\nTrying configuration. Confirm at the Netplan prompt to keep it.\n'
printf 'If your connection drops or you do not confirm, Netplan should roll back.\n'
if ! netplan try --timeout 60; then
  restore
  echo 'Netplan try failed or was not confirmed; original file restored.' >&2
  exit 1
fi
# Netplan confirmation only means the YAML was accepted. The actual WPA service
# and wireless association must also succeed. Retry a stale-socket repair once.
if command -v systemctl >/dev/null; then
  if ! systemctl is-active --quiet "$SERVICE"; then
    echo "WPA service is not active. Attempting one targeted repair and restart..."
    if repair_wpa_socket; then
      systemctl start "$SERVICE" 2>/dev/null || true
    else
      echo "WARNING: Could not automatically repair $IFACE without disrupting another process." >&2
    fi
  fi
fi
# Connection verification: distinguish WPA association, DHCP, and optional internet access.
# Do not change default routes, DNS, Ethernet, LTE, or other Wi-Fi interfaces.
verify_wifi() {
  local service_state link_out ip4 network_state i
  echo
  echo "=== Wi-Fi verification: $IFACE ==="
  echo '1. Netplan Wi-Fi configuration:'
  if IFACE="$IFACE" python3 - <<'PYVERIFY'
import glob,os,yaml
iface=os.environ['IFACE']
found=[]
for path in glob.glob('/etc/netplan/*.yaml'):
    try:
        with open(path) as f: cfg=yaml.safe_load(f) or {}
        if iface in (cfg.get('network',{}).get('wifis',{}) or {}): found.append(path)
    except Exception: pass
if found: print('   Configured:',', '.join(found))
else: print('   WARNING: no Netplan Wi-Fi entry for',iface)
PYVERIFY
  then :; else echo '   WARNING: Could not inspect Netplan YAML'; fi
  echo '2. WPA service:'
  service_state=$(systemctl is-active "$SERVICE" 2>/dev/null || true)
  echo "   $SERVICE: ${service_state:-unknown}"
  echo '3. Wireless association:'
  link_out=$(iw dev "$IFACE" link 2>&1 || true)
  printf '%s\n' "$link_out" | sed 's/^/   /'
  echo '4. IPv4 / DHCP:'
  for ((i=0;i<20;i++)); do
    ip4=$(ip -4 -o addr show dev "$IFACE" scope global 2>/dev/null | awk '{print $4}' | head -n 1)
    [[ -n "$ip4" ]] && break
    sleep 2
  done
  ip -br addr show dev "$IFACE" || true
  if [[ -n "$ip4" ]]; then
    echo "   IPv4 allocated: $ip4"
  else
    echo '   FAIL: No global IPv4 address assigned.'
  fi
  echo '5. networkd status:'
  if command -v networkctl >/dev/null; then
    networkctl status "$IFACE" --no-pager --lines=0 2>/dev/null | grep -E 'State:|Online state:|Network File:|Wi-Fi access point:|Address:|Gateway:' | head -n 12 || true
  fi
  echo '6. Selected interface DHCP/WPA diagnostics:'
  if [[ "$service_state" != active ]]; then
    echo '   WPA service is not active. Recent errors:'
    journalctl -b -u "$SERVICE" --no-pager -n 20 2>/dev/null | grep -Ei 'failed|error|control interface|association|authentication|status=' | tail -n 8 || true
  fi
  if [[ -z "$ip4" ]]; then
    echo '   Recent DHCP/networkd messages for this interface:'
    journalctl -b -u systemd-networkd --no-pager -n 150 2>/dev/null | grep -F "$IFACE" | grep -Ei 'dhcp|carrier|address|configur|fail|error' | tail -n 12 || true
  fi
  echo '7. Result:'
  if [[ "$service_state" != active ]]; then
    echo '   FAIL: WPA service is not running; Wi-Fi association cannot be trusted.'
  elif ! grep -q '^Connected to ' <<< "$link_out"; then
    echo '   FAIL: Adapter is not associated with an access point. Check WPA logs, SSID, password, security and band.'
  elif [[ -z "$ip4" ]]; then
    echo '   FAIL: Wi-Fi associated, but DHCP has not assigned IPv4. Check DHCP server and networkd configuration.'
  else
    echo '   PASS: WPA service active, Wi-Fi associated, and IPv4 address allocated.'
    echo '   Internet access was NOT tested (no default-route or DNS changes were made).'
  fi
}
verify_wifi
echo "Configuration: $TARGET"

# Persistent Wi-Fi naming is built directly into wifi-add.sh (no helper required).
# Only prepares a MAC-matched systemd .link rule and renames Netplan's Wi-Fi key.
# The live interface is never renamed; a reboot is required.
if [[ "$IFACE" != wlan0 ]]; then
  printf '\nOptional: assign this adapter a persistent wlan1/wlan2/... name (requires reboot).\n'
  read -r -p "Rename $IFACE after next reboot? [y/N]: " RENAME_ANSWER
  if [[ "$RENAME_ANSWER" =~ ^[Yy]$ ]]; then
    MAC=$(cat "/sys/class/net/$IFACE/address")
    if [[ ! "$MAC" =~ ^([[:xdigit:]]{2}:){5}[[:xdigit:]]{2}$ ]]; then
      echo 'WARNING: Adapter MAC address invalid; naming unchanged.' >&2
    else
      n=1
      while ip link show "wlan$n" >/dev/null 2>&1 || grep -RqlE "^[[:space:]]*wlan$n:" /etc/netplan --include='*.yaml' 2>/dev/null || grep -RqlE "^Name=wlan$n$" /etc/systemd/network /usr/lib/systemd/network 2>/dev/null; do n=$((n+1)); done
      NEW="wlan$n"
      read -r -p "New persistent name [$NEW]: " response
      NEW=${response:-$NEW}
      if [[ ! "$NEW" =~ ^wlan[1-9][0-9]*$ ]]; then
        echo 'WARNING: Name must be wlan1, wlan2, ...; naming unchanged.' >&2
      elif ip link show "$NEW" >/dev/null 2>&1; then
        echo "WARNING: $NEW is already in use; naming unchanged." >&2
      else
        export IFACE NEW MAC
        if ! python3 - <<'PYRENAME'
import os,glob,sys,subprocess,tempfile,shutil,datetime,yaml,re
old,new,mac=(os.environ[k] for k in ('IFACE','NEW','MAC'))
paths=sorted(glob.glob('/etc/netplan/*.yaml'))
loaded=[]
for path in paths:
    with open(path,encoding='utf-8') as f: cfg=yaml.safe_load(f) or {}
    loaded.append((path,cfg))
for path,cfg in loaded:
    wifi=cfg.get('network',{}).get('wifis',{}) or {}
    if new in wifi:
        sys.exit(f'ERROR: {new} is already configured in {path}')
entries=[(p,c) for p,c in loaded if old in (c.get('network',{}).get('wifis',{}) or {})]
if len(entries)>1:
    sys.exit('ERROR: Wi-Fi interface appears in multiple Netplan files')
if entries:
    path,cfg=entries[0]
    entry=cfg['network']['wifis'][old]
    if not isinstance(entry,dict): sys.exit('ERROR: invalid Wi-Fi entry')
    if entry.get('match') or entry.get('set-name'):
        sys.exit('ERROR: existing match/set-name in Wi-Fi entry; manual cleanup needed')
else:
    path='/etc/netplan/91-wifi-rename.yaml'
    if os.path.exists(path): sys.exit('ERROR: 91-wifi-rename.yaml already exists')
    cfg={'network':{'version':2,'wifis':{}}}
linkdir='/etc/systemd/network'
os.makedirs(linkdir,exist_ok=True)
link=os.path.join(linkdir,f'10-wifi-{mac.replace(":", "")}.link')
if os.path.lexists(link): sys.exit(f'ERROR: existing link policy {link}; inspect it first')
# Check for an existing conflicting policy assigning the same name or MAC.
for folder in ('/etc/systemd/network','/run/systemd/network'):
    for candidate in glob.glob(folder+'/*.link'):
        try: content=open(candidate,encoding='utf8').read()
        except OSError: continue
        if re.search(r'^Name\s*=\s*'+re.escape(new)+r'\s*$',content,re.M):
            sys.exit(f'ERROR: {new} already assigned by {candidate}')
        if re.search(r'^MACAddress\s*=\s*'+re.escape(mac)+r'\s*$',content,re.M|re.I):
            sys.exit(f'ERROR: MAC already has a link policy: {candidate}')
backup=None
if os.path.exists(path):
    backup=path+'.backup-wifi-rename-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    shutil.copy2(path,backup)
    print('Backup:',backup)
# Make all file changes transactional: netplan generate failure rolls back.
try:
    wifis=cfg['network']['wifis']
    if old in wifis: wifis[new]=wifis.pop(old)
    else: wifis[new]={'dhcp4':True,'optional':True}
    # Explicitly avoid match/set-name under networkd Wi-Fi.
    with open(path,'w',encoding='utf8') as f: yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
    os.chmod(path,0o600)
    with open(link,'x',encoding='utf8') as f:
        f.write(f'[Match]\nMACAddress={mac}\n\n[Link]\nNamePolicy=\nName={new}\n')
    os.chmod(link,0o644)
    result=subprocess.run(['netplan','generate'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('netplan generate failed: '+(result.stderr or result.stdout).strip())
except Exception as e:
    if backup: shutil.copy2(backup,path)
    else:
        try: os.unlink(path)
        except FileNotFoundError: pass
    try: os.unlink(link)
    except FileNotFoundError: pass
    subprocess.run(['netplan','generate'],capture_output=True)
    sys.exit('ERROR: '+str(e)+'\nOriginal configuration restored.')
print(f'Persistent rename prepared: {old} -> {new} (MAC {mac})')
print(f'Systemd link: {link}')
print(f'Netplan: {path}')
print('Reboot required. Current interfaces have NOT been modified; after reboot, run wifi-add.sh again if IPv4 is missing.')
print('After reboot: ip -br link; iw dev; networkctl status '+new)
PYRENAME
        then
          echo 'WARNING: Persistent naming failed; Wi-Fi configuration remains in place.' >&2
        fi
      fi
    fi
  fi
fi
