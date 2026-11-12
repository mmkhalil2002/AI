#!/usr/bin/env bash
# Stop the local Ollama server without uninstalling Ollama or deleting models.
set -Eeuo pipefail

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files ollama.service --no-legend 2>/dev/null | grep -q '^ollama.service'; then
  echo '[INFO] Stopping Ollama systemd service...'
  if [[ $EUID -eq 0 ]]; then
    systemctl stop ollama
  else
    sudo systemctl stop ollama
  fi
fi

# Stop a manually launched `ollama serve`, if present.
if pgrep -f '(^|/)ollama serve($| )' >/dev/null 2>&1; then
  echo '[INFO] Stopping manually launched Ollama server...'
  pkill -f '(^|/)ollama serve($| )' || true
fi

if command -v curl >/dev/null 2>&1 && curl --silent --max-time 2 --fail http://127.0.0.1:11434/ >/dev/null; then
  echo '[WARNING] Ollama is still responding on port 11434. Check for another Ollama process.'
  exit 1
fi

echo '[SUCCESS] Ollama stopped. Installed models are unchanged.'
echo '[NOTE] If Ollama is enabled at boot, it will start again after reboot.'
