#!/usr/bin/env bash
# Start Ollama with local-only or LAN listening. Does not change firewall rules.
set -Eeuo pipefail
MODE=local; PULL=0
for arg in "$@"; do
  case "$arg" in
    --local) MODE=local ;;
    --all-ips) MODE=all ;;
    --pull-models) PULL=1 ;;
    *) echo "Usage: $0 [--local|--all-ips] [--pull-models]" >&2; exit 2 ;;
  esac
done
if [[ "$MODE" == all ]]; then HOST=0.0.0.0:11434; else HOST=127.0.0.1:11434; fi
API=http://127.0.0.1:11434
command -v ollama >/dev/null || { echo '[ERROR] Install Ollama first.' >&2; exit 1; }
command -v curl >/dev/null || { echo '[ERROR] Install curl first.' >&2; exit 1; }
if command -v systemctl >/dev/null && systemctl cat ollama.service >/dev/null 2>&1; then
  echo "[INFO] Configuring systemd Ollama service: $HOST"
  if [[ "$(id -u)" == 0 ]]; then SUDO=(); else SUDO=(sudo); fi
  "${SUDO[@]}" mkdir -p /etc/systemd/system/ollama.service.d
  printf '[Service]\nEnvironment="OLLAMA_HOST=%s"\n' "$HOST" | "${SUDO[@]}" tee /etc/systemd/system/ollama.service.d/10-listen.conf >/dev/null
  "${SUDO[@]}" systemctl daemon-reload
  "${SUDO[@]}" systemctl restart ollama
else
  if curl -fsS --max-time 2 "$API/api/tags" >/dev/null 2>&1; then
    echo '[WARNING] Ollama is already running outside systemd. Cannot safely change its bind address.' >&2
    echo '[ACTION] Stop the existing Ollama process, then rerun this script with the desired option.' >&2
    exit 1
  fi
  echo "[INFO] Starting Ollama with OLLAMA_HOST=$HOST"
  OLLAMA_HOST="$HOST" nohup ollama serve >"$HOME/ollama.log" 2>&1 </dev/null &
fi
READY=0
for ((i=0; i<30; i++)); do
  if curl -fsS --max-time 2 "$API/api/tags" >/dev/null 2>&1; then READY=1; break; fi
  sleep 1
done
(( READY )) || { echo '[ERROR] Ollama did not become ready. Check service status or ~/ollama.log.' >&2; exit 1; }
echo "[SUCCESS] Ollama API ready. Requested bind: $HOST"
echo '[INFO] Verify actual listening address with: ss -lntp | grep 11434'
ollama list
if (( PULL )); then
  ollama pull medgemma1.5
  ollama pull qwen3:8b
fi
if [[ "$MODE" == all ]]; then echo '[SECURITY] Restrict TCP 11434 to trusted LAN hosts using your firewall; never expose publicly.'; fi
