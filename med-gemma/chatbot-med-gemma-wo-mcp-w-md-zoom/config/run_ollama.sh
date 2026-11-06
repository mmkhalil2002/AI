#!/usr/bin/env bash
# ============================================================================
# run_ollama.sh — Start, inspect, and test an Ollama server on Ubuntu.
#
# USAGE (no arguments prints this help; it does NOT start Ollama):
#   ./run_ollama.sh                 Show help
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

usage() {
  sed -n '2,/^# ============================================================================/p' "$0" | sed 's/^# \{0,1\}//'
}
info() { printf '[INFO] %s\n' "$*"; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
api_ready() { curl -fsS --max-time 3 "$API/api/tags" >/dev/null 2>&1; }

# Parse all options first so they can be combined in any order.
if (( $# == 0 )); then usage; exit 0; fi
while (( $# > 0 )); do
  case "$1" in
    -h|-help|--help) usage; exit 0 ;;
    -r|-remote|-cr|-connect_remotely|--all-ips)
      [[ -z "$MODE" || "$MODE" == remote ]] || fail 'Cannot combine remote and local modes.'
      MODE=remote ;;
    -l|-local|--local)
      [[ -z "$MODE" || "$MODE" == local ]] || fail 'Cannot combine remote and local modes.'
      MODE=local ;;
    -t|-test|--test) DO_TEST=true ;;
    -s|-status|--status) DO_STATUS=true ;;
    -m|-models|--models) DO_MODELS=true ;;
    -pull|--pull) (( $# >= 2 )) || fail "$1 requires a model name."; PULL_MODEL="$2"; shift ;;
    -model|--model) (( $# >= 2 )) || fail "$1 requires a model name."; MODEL="$2"; shift ;;
    *) fail "Unknown option: $1 (try -help)" ;;
  esac
  shift
done

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
