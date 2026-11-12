#!/usr/bin/env bash
# ============================================================================
# run_ollama.sh — Start, inspect, and test an Ollama server on Ubuntu.
#
# USAGE (no arguments displays an interactive menu):
#   ./run_ollama.sh                 Interactive menu (choose 1-13)
#   ./run_ollama.sh -h              Show help
#   ./run_ollama.sh -remote         Start for trusted remote/LAN clients
#   ./run_ollama.sh -local          Start for localhost clients only
#   ./run_ollama.sh -remote -test   Start remotely and test MedGemma
#   ./run_ollama.sh -t              Test an already-running server
#   ./run_ollama.sh -s              Show API/listening status
#   ./run_ollama.sh -m              List downloaded models
#   ./run_ollama.sh -pull qwen3:8b  Download a model (requires running server)
#
# OPTIONS:
#   -h, -help, --help                  Display this help
#   -r, -remote, -cr, -connect_remotely, --all-ips  Bind to 0.0.0.0:11434 (remote access)
#   -l, -local, --local               Bind to 127.0.0.1:11434 (local only)
#   -t, -test, --test                 Ask MedGemma "Hello" and show its reply
#   -s, -status, --status             Check API and listening address
#   -m, -models, --models             List installed Ollama models
#   -pull, --pull MODEL               Download MODEL via Ollama API
#   -model, --model MODEL             Model to use with -test (default: medgemma1.5)
#   -autostart-local                   Enable Ollama at boot (local only)
#   -autostart-remote                  Enable Ollama at boot (trusted LAN)
#   -autostart-off                     Disable automatic startup
#   -autostart-status                  Show boot service status
#
# NOTES:
#   * The server is started in the background; logs go to
#     ${OLLAMA_LOG:-/tmp/ollama-serve.log}.
#   * If Ollama is already running, the script will not change its bind address.
#     Stop it using ./unrun_ollama.sh, then restart with -remote or -local.
#   * Remote access is unauthenticated by default. Limit TCP 11434 to trusted
#     hosts with a firewall. Never expose this port directly to the Internet.
#   * OLLAMA_HOST can be overridden by the selected start mode. OLLAMA_PORT,
#     OLLAMA_MODEL, and OLLAMA_LOG can be set in the environment.
#   * This script does not install Ollama or delete models.
# ============================================================================
set -Eeuo pipefail

PORT="${OLLAMA_PORT:-11434}"
MODEL="${OLLAMA_MODEL:-medgemma1.5}"
LOG_FILE="${OLLAMA_LOG:-/tmp/ollama-serve.log}"
API="http://127.0.0.1:${PORT}"
MODE=""
DO_TEST=false
DO_STATUS=false
DO_MODELS=false
PULL_MODEL=""
BOOT_ACTION=""

usage() {
  sed -n '2,/^# ============================================================================/p' "$0" | sed 's/^# \{0,1\}//'
}
info() { printf '[INFO] %s\n' "$*"; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
api_ready() { curl -fsS --max-time 3 "$API/api/tags" >/dev/null 2>&1; }

# Show an interactive menu only when no command-line options are supplied.
# The selected menu item is converted into normal command-line arguments,
# so all actions use the same tested execution path as -remote, -test, etc.
show_menu() {
  local choice model_name
  while true; do
    cat <<'MENU'
================================================
          OLLAMA SERVER CONTROL MENU
================================================
1. Start Ollama - Remote Access
2. Start Ollama - Local Access
3. Test MedGemma (Send Hello)
4. Start Remote Access + Test MedGemma
5. Check Ollama Status
6. List Installed Models
7. Download a Model
8. Display Help
9. Exit
10. Enable Auto-start at Reboot - Local
11. Enable Auto-start at Reboot - Remote LAN
12. Disable Auto-start at Reboot
13. Check Auto-start Service Status
================================================
MENU
    read -r -p 'Enter your selection [1-13]: ' choice || return 0
    case "$choice" in
      1) set -- -remote; break ;;
      2) set -- -local; break ;;
      3) set -- -test; break ;;
      4) set -- -remote -test; break ;;
      5) set -- -status; break ;;
      6) set -- -models; break ;;
      7)
        read -r -p 'Enter model name (e.g. medgemma1.5): ' model_name || return 0
        if [[ -z "$model_name" ]]; then
          printf '[WARNING] Model name cannot be empty.\n'
          continue
        fi
        set -- -pull "$model_name"
        break ;;
      8) usage; continue ;;
      10) set -- -autostart-local; break ;;
      11) set -- -autostart-remote; break ;;
      12) set -- -autostart-off; break ;;
      13) set -- -autostart-status; break ;;
      9) printf 'Exiting without changes.\n'; return 0 ;;
      *) printf '[WARNING] Please select a number from 1 to 13.\n' ;;
    esac
  done
  # Return the selected arguments through the global array for the parser.
  MENU_ARGS=("$@")
}

# Parse all options first so they can be combined in any order.
if (( $# == 0 )); then
  MENU_ARGS=()
  show_menu
  if (( ${#MENU_ARGS[@]} == 0 )); then exit 0; fi
  set -- "${MENU_ARGS[@]}"
fi
while (( $# > 0 )); do
  case "$1" in
    -h|-help|--help) usage; exit 0 ;;
    -r|-remote|-cr|-connect_remotely|--all-ips)
      [[ -z "$MODE" || "$MODE" == remote ]] || fail 'Cannot combine remote and local modes.'
      MODE=remote ;;
    -l|-local|--local)
      [[ -z "$MODE" || "$MODE" == local ]] || fail 'Cannot combine remote and local modes.'
      MODE=local ;;
    -autostart-local|--autostart-local) BOOT_ACTION=local ;;
    -autostart-remote|--autostart-remote) BOOT_ACTION=remote ;;
    -autostart-off|--autostart-off) BOOT_ACTION=off ;;
    -autostart-status|--autostart-status) BOOT_ACTION=status ;;
    -t|-test|--test) DO_TEST=true ;;
    -s|-status|--status) DO_STATUS=true ;;
    -m|-models|--models) DO_MODELS=true ;;
    -pull|--pull) (( $# >= 2 )) || fail "$1 requires a model name."; PULL_MODEL="$2"; shift ;;
    -model|--model) (( $# >= 2 )) || fail "$1 requires a model name."; MODEL="$2"; shift ;;
    *) fail "Unknown option: $1 (try -help)" ;;
  esac
  shift
done

# Configure persistent boot startup through systemd. The service uses the current
# account and its existing Ollama model directory, avoiding a second empty model store.
# This action may request sudo. It does not download models or open firewall ports.
configure_autostart() {
  local action="$1" binary service_user service_home bind unit_file
  command -v systemctl >/dev/null 2>&1 || fail 'systemd is required for auto-start.'
  command -v sudo >/dev/null 2>&1 || [[ "$EUID" -eq 0 ]] || fail 'sudo is required.'
  if [[ "$action" == status ]]; then
    systemctl is-enabled ollama.service 2>/dev/null || true
    systemctl is-active ollama.service 2>/dev/null || true
    systemctl status ollama.service --no-pager --lines=8 || true
    return 0
  fi
  if [[ "$action" == off ]]; then
    if systemctl cat ollama.service >/dev/null 2>&1; then
      sudo systemctl disable ollama.service || fail 'Could not disable Ollama service.'
      info 'Automatic startup disabled. Existing running server was not stopped.'
    else
      info 'No Ollama systemd service is installed.'
    fi
    return 0
  fi
  binary="$(command -v ollama)" || fail 'Ollama not installed; run init.sh.'
  service_user="$(id -un)"
  service_home="$HOME"
  # systemd's unit syntax is not shell syntax: reject problematic paths/usernames.
  [[ "$binary" =~ ^/[a-zA-Z0-9_./+-]+$ && "$service_home" =~ ^/[a-zA-Z0-9_./+-]+$ && "$service_user" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Unsupported special characters in Ollama path, username, or home.'
  if [[ "$action" == remote ]]; then bind="0.0.0.0:$PORT"; else bind="127.0.0.1:$PORT"; fi
  [[ "$PORT" =~ ^[0-9]+$ ]] || fail 'OLLAMA_PORT must be numeric.'
  if [[ "$action" == remote ]]; then
    printf '[SECURITY] Remote Ollama has NO built-in authentication. Restrict TCP %s to trusted LAN hosts.\n' "$PORT"
    local confirm
    read -r -p 'Enable remote LAN access at every reboot? [y/N] ' confirm || return 1
    [[ "$confirm" =~ ^[Yy]$ ]] || { info 'Cancelled.'; return 0; }
  fi
  # Stop the existing systemd-managed server, if any, before replacing the unit.
  # An independently started 'ollama serve' may also be using the same port.
  unit_file="$(mktemp)"
  cat >"$unit_file" <<EOF
[Unit]
Description=Ollama AI Model Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
Environment="HOME=$service_home"
Environment="OLLAMA_HOST=$bind"
ExecStart=$binary serve
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  if [[ "$EUID" -eq 0 ]]; then
    install -m 644 "$unit_file" /etc/systemd/system/ollama.service
    systemctl daemon-reload
    systemctl enable ollama.service
  else
    sudo install -m 644 "$unit_file" /etc/systemd/system/ollama.service
    sudo systemctl daemon-reload
    sudo systemctl enable ollama.service
  fi
  rm -f "$unit_file"
  # A running manually started server could occupy the port. Avoid stopping it
  # unexpectedly: the boot service is enabled, but the user should switch cleanly.
  if api_ready && ! systemctl is-active --quiet ollama.service; then
    info 'Auto-start ENABLED for next reboot; an existing manual Ollama server is running.'
    info 'To activate the new mode now, stop the manual server first, then: sudo systemctl start ollama'
  else
    if [[ "$EUID" -eq 0 ]]; then systemctl restart ollama.service; else sudo systemctl restart ollama.service; fi
    systemctl is-active --quiet ollama.service || fail 'Service did not start. Check: sudo journalctl -u ollama -n 50'
    info "Auto-start ENABLED and service active ($bind)."
  fi
}

if [[ -n "$BOOT_ACTION" ]]; then
  configure_autostart "$BOOT_ACTION"
fi

# Start the daemon only if -remote or -local was explicitly requested.
if [[ -n "$MODE" ]]; then
  command -v ollama >/dev/null 2>&1 || fail 'ollama command not found; run init.sh first.'
  command -v curl >/dev/null 2>&1 || fail 'curl is required.'
  if api_ready; then
    info 'Ollama API is already running. Existing listening address was NOT changed.'
    info 'To change local/remote mode, stop the existing server first.'
  else
    if [[ "$MODE" == remote ]]; then HOST="0.0.0.0:$PORT"; else HOST="127.0.0.1:$PORT"; fi
    info "Starting Ollama with OLLAMA_HOST=$HOST"
    # nohup keeps Ollama running after the terminal exits. Redirect logs.
    nohup env OLLAMA_HOST="$HOST" ollama serve >"$LOG_FILE" 2>&1 </dev/null &
    SERVER_PID=$!
    ready=false
    for (( i=0; i<30; i++ )); do
      if api_ready; then ready=true; break; fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if [[ "$ready" != true ]]; then
      printf '[ERROR] Ollama failed to become ready. Recent log output:\n' >&2
      tail -n 30 "$LOG_FILE" >&2 || true
      exit 1
    fi
    info "Ollama API ready at $API (requested bind: $HOST)"
    info "Server log: $LOG_FILE"
    if [[ "$MODE" == remote ]]; then
      printf '[SECURITY] Allow TCP %s only from trusted LAN hosts.\n' "$PORT"
    fi
  fi
fi

# Status works without starting the daemon.
if [[ "$DO_STATUS" == true ]]; then
  if api_ready; then info "Ollama API responding at $API"; else info "Ollama API not responding at $API"; fi
  if command -v ss >/dev/null 2>&1; then
    ss -lntp | grep -E ":${PORT}[[:space:]]" || true
  fi
fi

# List installed models using the existing server.
if [[ "$DO_MODELS" == true ]]; then
  api_ready || fail "Ollama API is not responding at $API; start with -local or -remote."
  ollama list
fi

# Pull models only on explicit request. This may download several GB.
if [[ -n "$PULL_MODEL" ]]; then
  api_ready || fail "Ollama API is not responding at $API; start with -local or -remote."
  info "Downloading model: $PULL_MODEL"
  ollama pull "$PULL_MODEL"
fi

# Generate a response, printing only text rather than raw JSON.
if [[ "$DO_TEST" == true ]]; then
  api_ready || fail "Ollama API is not responding at $API; start with -local or -remote."
  command -v python3 >/dev/null 2>&1 || fail 'python3 is required for -test.'
  if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "${MODEL%:latest}:latest" && ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL"; then
    printf '[WARNING] Model %s is missing.\n' "$MODEL"
    read -r -p "Download $MODEL now? [y/N] " confirm || exit 1
    [[ "$confirm" =~ ^[Yy]$ ]] || fail 'Test cancelled: model not installed.'
    ollama pull "$MODEL" || fail 'Model download failed.'
  fi
  info "Sending 'Hello' to model: $MODEL"
  # Python builds valid JSON, so custom model names cannot break shell quoting.
  REQUEST=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":"Hello","stream":False}))' "$MODEL")
  RESPONSE=$(curl -fsS --max-time 600 -H 'Content-Type: application/json' -d "$REQUEST" "$API/api/generate") || fail 'Model inference request failed.'
  printf '%s' "$RESPONSE" | python3 -c '
import json,sys
try:
    data=json.load(sys.stdin)
except Exception as e:
    sys.exit("Invalid Ollama JSON response: " + str(e))
if data.get("error"):
    sys.exit("Ollama error: " + str(data["error"]))
print("[RESPONSE]")
print(data.get("response", "(empty response)"))
print("\n[STATS] Input tokens: {}, output tokens: {}".format(data.get("prompt_eval_count", "?"), data.get("eval_count", "?")))
'
fi
