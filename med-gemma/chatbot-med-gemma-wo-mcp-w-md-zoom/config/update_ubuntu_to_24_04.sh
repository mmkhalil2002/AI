#!/usr/bin/env bash
# Upgrade Ubuntu Server through sequential supported LTS releases.
# Usage: sudo bash update_ubuntu_to_latest_version.sh [--auto-reboot] [--resume] [--status] [--diagnose]
# Target: Ubuntu 24.04 LTS. From 20.04, upgrade to 22.04, reboot, then 24.04.
# --diagnose: display OS/kernel, upgrade availability, service state, and recent logs.
# --auto-reboot: enable automatic reboots and systemd continuation between releases.
# Without --auto-reboot, run this script again after each manual reboot.
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
SCRIPT_NAME=update_ubuntu_to_latest_version.sh
STATE_DIR=/var/lib/ubuntu-lts-upgrader
LOG_DIR=/var/log/ubuntu-lts-upgrader
INSTALL_PATH=/usr/local/sbin/$SCRIPT_NAME
UNIT=/etc/systemd/system/ubuntu-lts-upgrader.service
AUTO=0
RESUME=0
STATUS=0
DIAGNOSE=0
TARGET_VERSION=24.04
for arg in "$@"; do
  case "$arg" in
    --auto-reboot) AUTO=1;;
    --resume) RESUME=1;;
    --status) STATUS=1;;
    --diagnose) DIAGNOSE=1;;
    -h|--help) echo "Usage: sudo bash $SCRIPT_NAME [--auto-reboot|--resume|--status|--diagnose]"; exit 0;;
    *) echo "Unknown argument: $arg" >&2; exit 2;;
  esac
done
(( EUID == 0 )) || { echo 'Run with sudo.' >&2; exit 1; }
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 9>"$STATE_DIR/lock"
flock -n 9 || { echo 'Another updater instance is running.' >&2; exit 1; }
LOG="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; echo "ERROR: line $LINENO, exit code $rc. Log: $LOG" >&2' ERR
say(){ echo "[$(date -u +%FT%TZ)] $*"; }
# Print diagnostic information without making any package or OS changes.
# The upgrade availability check may contact Ubuntu release metadata services.
if (( DIAGNOSE )); then
  echo '=== Ubuntu release ==='
  grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true
  echo "Running kernel: $(uname -r)"
  echo '=== Upgrade status ==='
  echo "Automatic mode: $(cat "$STATE_DIR/auto" 2>/dev/null || echo off)"
  if [[ -f "$STATE_DIR/pending_reboot" ]]; then
    old_boot=$(cat "$STATE_DIR/pending_reboot")
    current_boot=$(cat /proc/sys/kernel/random/boot_id)
    if [[ "$old_boot" == "$current_boot" ]]; then
      echo 'Pending reboot: YES (reboot before continuing)'
    else
      echo 'Pending reboot: previous reboot completed (rerun script)'
    fi
  else
    echo 'Pending reboot: no'
  fi
  echo "Free disk space on /: $(df -hP / | awk 'NR==2 {print $4}')"
  echo '=== Available LTS release ==='
  if command -v do-release-upgrade >/dev/null 2>&1; then
    do-release-upgrade -c || echo 'Release availability check failed; review networking and logs.'
  else
    echo 'do-release-upgrade is not installed.'
  fi
  echo '=== Recent upgrade logs ==='
  find "$LOG_DIR" -maxdepth 1 -type f -name 'run-*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 3 | cut -d' ' -f2-
  echo '=== Release upgrader service ==='
  systemctl is-enabled ubuntu-lts-upgrader.service 2>/dev/null || true
  systemctl is-active ubuntu-lts-upgrader.service 2>/dev/null || true
  echo '=== Next action ==='
  echo 'If 24.04 is offered, backups are verified, and no upgrade is running:'
  echo "  sudo bash $SCRIPT_NAME"
  echo 'If a reboot is pending: sudo reboot; then rerun the script.'
  exit 0
fi
if (( STATUS )); then
  cat /etc/os-release | grep -E '^(PRETTY_NAME|VERSION_ID)='
  echo "Automatic mode: $(cat "$STATE_DIR/auto" 2>/dev/null || echo off)"
  echo "Pending reboot: $(cat "$STATE_DIR/pending_reboot" 2>/dev/null || echo no)"
  echo "Logs: $LOG_DIR"
  echo "Target: Ubuntu $TARGET_VERSION LTS"
  echo "For detailed checks: sudo bash $SCRIPT_NAME --diagnose"
  exit 0
fi
source /etc/os-release
[[ "${ID:-}" == ubuntu ]] || { say 'Only Ubuntu is supported.'; exit 1; }
if [[ "$VERSION_ID" == "$TARGET_VERSION" ]]; then
  say "Target Ubuntu $TARGET_VERSION LTS is already installed; no release upgrade needed."
  if [[ -f "$STATE_DIR/auto" ]]; then
    systemctl disable ubuntu-lts-upgrader.service 2>/dev/null || true
    rm -f "$STATE_DIR/auto"
  fi
  exit 0
fi
if [[ "$VERSION_ID" != 20.04 && "$VERSION_ID" != 22.04 ]]; then
  say "This script supports upgrading 20.04 -> 22.04 -> $TARGET_VERSION only; stopping."
  exit 1
fi
[[ $(uname -m) == x86_64 || $(uname -m) == aarch64 ]] || { say 'Unsupported CPU architecture.'; exit 1; }
if (( AUTO )); then echo on > "$STATE_DIR/auto"; fi
if [[ $(cat "$STATE_DIR/auto" 2>/dev/null || echo off) == on ]]; then AUTO=1; fi
if (( AUTO )) && (( ! RESUME )); then
  echo 'WARNING: Automatic mode may reboot and interrupt SSH. Ensure console access and external backups.'
  read -r -p 'Type UPGRADE to enable automatic sequential LTS upgrades: ' reply
  [[ "$reply" == UPGRADE ]] || exit 1
fi
# Do not attempt to upgrade on a nearly full root filesystem.
free_kb=$(df -Pk / | awk 'NR==2 {print $4}')
(( free_kb >= 8*1024*1024 )) || { say 'At least 8 GiB free on / is required (more may be needed).'; exit 1; }
# Backups must exist off-host; a local backup alone does not protect from disk failure.
if [[ ! -f "$STATE_DIR/backup_confirmed" ]]; then
  if (( RESUME )); then say 'Backup confirmation missing; stopping.'; exit 1; fi
  echo 'Ensure you have a verified external backup of your project, Ollama models and configuration.'
  read -r -p 'Type BACKED-UP to confirm: ' reply
  [[ "$reply" == BACKED-UP ]] || exit 1
  touch "$STATE_DIR/backup_confirmed"
fi
# Install a persistent copy and a resume service before any release change.
if (( AUTO )); then
  if [[ $(readlink -f "$0") != "$INSTALL_PATH" ]]; then install -m 0755 "$(readlink -f "$0")" "$INSTALL_PATH"; fi
  cat > "$UNIT" <<UNITFILE
[Unit]
Description=Resume sequential Ubuntu LTS upgrade
After=network-online.target
Wants=network-online.target
ConditionPathExists=$STATE_DIR/auto
[Service]
Type=oneshot
ExecStart=/bin/bash $INSTALL_PATH --resume
TimeoutStartSec=infinity
[Install]
WantedBy=multi-user.target
UNITFILE
  systemctl daemon-reload
  systemctl enable ubuntu-lts-upgrader.service
fi
# Do not resume until the reboot actually happened.
if [[ -f "$STATE_DIR/pending_reboot" ]]; then
  old_boot=$(cat "$STATE_DIR/pending_reboot")
  current_boot=$(cat /proc/sys/kernel/random/boot_id)
  if [[ "$old_boot" == "$current_boot" ]]; then
    say 'A reboot is required before continuing.'
    if (( AUTO )); then
      say 'Rebooting in 30 seconds; Ctrl+C to cancel.'
      sleep 30
      systemctl reboot
    fi
    exit 0
  fi
  rm -f "$STATE_DIR/pending_reboot"
fi
# A release upgrade may leave partially configured packages; stop rather than masking errors.
if pgrep -af 'do-release-upgrade|DistUpgrade' | grep -v "$$" | grep -q .; then
  say 'Another release upgrade may be running. Check processes before retrying.'; exit 1
fi
say "Current Ubuntu: ${VERSION_ID}; log: $LOG"
# Run only if the system has working package sources and DNS.
getent hosts archive.ubuntu.com >/dev/null || { say 'DNS/network unavailable.'; exit 1; }
# Use standard Ubuntu release-upgrader policy, and never force development releases.
apt-get update
apt-get -y install ubuntu-release-upgrader-core update-manager-core ca-certificates
if [[ -f /etc/update-manager/release-upgrades ]]; then
  sed -i -E 's/^Prompt=.*/Prompt=lts/' /etc/update-manager/release-upgrades
fi
# Stop for dpkg inconsistencies; repair explicitly if needed.
dpkg --audit
apt-get -y upgrade
apt-get -y full-upgrade
if [[ -e /var/run/reboot-required ]]; then
  say 'Updates require a reboot before the release upgrade.'
  cat /proc/sys/kernel/random/boot_id > "$STATE_DIR/pending_reboot"
  if (( AUTO )); then sleep 15; systemctl reboot; fi
  exit 0
fi
say 'Checking for the next supported LTS release.'
set +e
check_output=$(do-release-upgrade -c 2>&1)
check_rc=$?
set -e
printf '%s\n' "$check_output"
if grep -qi 'No new release found' <<< "$check_output"; then
  say "No next LTS release offered yet; target remains Ubuntu $TARGET_VERSION."
  say "Run sudo bash $SCRIPT_NAME --diagnose to inspect availability and logs."
  if (( AUTO )); then
    systemctl disable ubuntu-lts-upgrader.service
    rm -f "$STATE_DIR/auto"
  fi
  exit 0
fi
if (( check_rc != 0 )); then
  say 'Release availability check failed. Stopping safely.'; exit 1
fi
if ! grep -qiE 'New release|available' <<< "$check_output"; then
  say 'Could not confirm an upgrade target. Stopping safely.'; exit 1
fi
say 'Starting release upgrade. Ubuntu may require interactive decisions.'
# The Ubuntu upgrader controls package/configuration decisions; do not override them.
# -f DistUpgradeViewNonInteractive is intentionally NOT used: unattended upgrades
# can make irreversible config decisions. Use a console or persistent tmux session.
if (( AUTO )); then
  say 'Auto-reboot mode only automates reboot/resume; the release upgrade itself can still prompt.'
fi
# Do not use -d (development releases).
do-release-upgrade
say 'Release upgrade command returned successfully.'
cat /proc/sys/kernel/random/boot_id > "$STATE_DIR/pending_reboot"
if (( AUTO )); then
  say 'Rebooting to continue to the next supported LTS release in 30 seconds.'
  sleep 30
  systemctl reboot
else
  say "Reboot now with sudo reboot, then rerun this script to continue toward Ubuntu $TARGET_VERSION."
fi
