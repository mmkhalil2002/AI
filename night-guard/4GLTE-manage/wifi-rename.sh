#!/usr/bin/env bash
# Assign a persistent wlan1/wlan2/... name to a detected Wi-Fi adapter.
# Uses a MAC-matched systemd .link file (NOT Netplan wifi match/set-name,
# which is unsupported by the networkd Wi-Fi backend on Ubuntu 24.04).
# Usage: sudo ./wifi-rename.sh [current-interface] [wlanN]
# Only prepares files for next reboot; never renames a live interface.
set -Eeuo pipefail
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo 'Usage: sudo ./wifi-rename.sh [current-interface] [wlanN]'
  echo 'Assigns persistent wlanN name by MAC, effective after reboot.'
  exit 0
fi
(( EUID == 0 )) || { echo 'Run as root (sudo)' >&2; exit 1; }
for cmd in iw ip python3 netplan; do command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }; done
python3 -c 'import yaml' || { echo 'Install python3-yaml' >&2; exit 1; }
mapfile -t adapters < <(iw dev | awk '$1=="Interface"{print $2}')
((${#adapters[@]})) || { echo 'No Wi-Fi interfaces found' >&2; exit 1; }
if [[ $# -ge 1 ]]; then IFACE=$1; else
  echo 'Detected Wi-Fi interfaces:'
  for i in "${!adapters[@]}"; do printf '%s) %s\n' "$((i+1))" "${adapters[$i]}"; done
  read -r -p 'Select interface number: ' selection
  [[ "$selection" =~ ^[0-9]+$ ]] && ((10#$selection>=1 && 10#$selection<=${#adapters[@]})) || { echo 'Invalid selection' >&2; exit 2; }
  IFACE=${adapters[$((10#$selection-1))]}
fi
printf '%s\n' "${adapters[@]}" | grep -Fxq -- "$IFACE" || { echo 'Not a detected Wi-Fi interface' >&2; exit 2; }
[[ "$IFACE" != wlan0 ]] || { echo 'Preserving onboard wlan0' >&2; exit 2; }
MAC=$(cat "/sys/class/net/$IFACE/address")
[[ "$MAC" =~ ^([[:xdigit:]]{2}:){5}[[:xdigit:]]{2}$ ]] || { echo 'Invalid MAC' >&2; exit 2; }
if [[ $# -ge 2 ]]; then NEW=$2; else
  n=1
  while ip link show "wlan$n" >/dev/null 2>&1 || grep -RqlE "^[[:space:]]*wlan$n:" /etc/netplan --include='*.yaml' 2>/dev/null || grep -RqlE "^Name=wlan$n$" /etc/systemd/network /usr/lib/systemd/network 2>/dev/null; do n=$((n+1)); done
  NEW="wlan$n"
  read -r -p "New persistent name [$NEW]: " response
  NEW=${response:-$NEW}
fi
[[ "$NEW" =~ ^wlan[1-9][0-9]*$ ]] || { echo 'Name must be wlan1, wlan2, ...' >&2; exit 2; }
[[ "$IFACE" != "$NEW" ]] || { echo 'Already named that'; exit 0; }
! ip link show "$NEW" >/dev/null 2>&1 || { echo 'Name already in use' >&2; exit 2; }
export IFACE NEW MAC
python3 - <<'PY'
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
print('Reboot required. Current interfaces have NOT been modified.')
print('After reboot: ip -br link; iw dev; networkctl status '+new)
PY
