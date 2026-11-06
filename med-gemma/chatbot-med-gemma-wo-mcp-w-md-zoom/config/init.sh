#!/usr/bin/env bash
# Run from config/ or anywhere. Does NOT launch Codex, Ollama server or Python GUI.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "=== MedGemma Ubuntu INIT ==="
echo "Project: $ROOT"
trap 'echo "[ERROR] Initialization failed at line $LINENO." >&2' ERR
command -v apt-get >/dev/null || { echo '[ERROR] Ubuntu/Debian apt-get required.' >&2; exit 1; }
APT=(apt-get)
if (( EUID != 0 )); then
  command -v sudo >/dev/null || { echo "[ERROR] sudo required for non-root user." >&2; exit 1; }
  APT=(sudo apt-get)
fi
"${APT[@]}" update
"${APT[@]}" install -y python3 python3-pip curl ca-certificates zstd python3-cairo python3-testresources
node_major=0
if command -v node >/dev/null 2>&1; then
  node_major="$(node -p 'process.versions.node.split(".")[0]')"
fi
if ! command -v npm >/dev/null 2>&1 || (( node_major < 20 )); then
  echo '[INFO] Installing Node.js 22 from NodeSource...'
  "${APT[@]}" install -y gnupg
  curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/medgemma_nodesource_setup.sh
  if (( EUID == 0 )); then bash /tmp/medgemma_nodesource_setup.sh; else sudo -E bash /tmp/medgemma_nodesource_setup.sh; fi
  "${APT[@]}" install -y nodejs
fi
node --version
npm --version

# Writable npm prefix and persistent bash PATH.
NPM_PREFIX="$HOME/.local/npm"
mkdir -p "$NPM_PREFIX/bin"
npm config set prefix "$NPM_PREFIX" --location=user
export PATH="$NPM_PREFIX/bin:$PATH"
path_line='export PATH="$HOME/.local/npm/bin:$PATH"'
touch "$HOME/.bashrc"
grep -Fqx "$path_line" "$HOME/.bashrc" || printf '\n# npm user executables\n%s\n' "$path_line" >> "$HOME/.bashrc"
echo '[INFO] Installing/updating Codex CLI...'
npm install -g @openai/codex
command -v codex >/dev/null || { echo '[ERROR] Codex not found on PATH.' >&2; exit 1; }
codex --version

# No virtual environment. Install Python packages in user site packages.
# Ubuntu 24.04+ enforces PEP 668: --break-system-packages permits --user installs.
# This does not modify apt-managed Python packages; it installs in ~/.local.
if [[ -d "$ROOT/.venv" ]]; then
  echo '[INFO] Removing project .venv (requested no-venv setup)...'
  rm -rf -- "$ROOT/.venv"
fi
VPY="$(command -v python3)"
PIP_ARGS=(--user)
if "$VPY" -m pip install --help | grep -q -- --break-system-packages; then
  PIP_ARGS+=(--break-system-packages)
fi
if [[ ! -f "$ROOT/requirements.txt" ]]; then
  if [[ ! -d "$ROOT/python" ]] || ! find "$ROOT/python" -type f -name '*.py' -print -quit | grep -q .; then
    echo '[ERROR] Need requirements.txt or Python sources in python/.' >&2
    exit 1
  fi
  echo '[INFO] Installing pipreqs in user Python packages...'
  "$VPY" -m pip install "${PIP_ARGS[@]}" pipreqs
  echo '[INFO] Generating requirements.txt...'
  "$VPY" -m pipreqs.pipreqs "$ROOT/python" \
    --savepath "$ROOT/requirements.txt" --mode no-pin --use-local
  [[ -f "$ROOT/requirements.txt" ]] || { echo '[ERROR] requirements.txt not created.' >&2; exit 1; }
fi
echo '[INFO] Installing application dependencies in user Python packages...'
"$VPY" -m pip install "${PIP_ARGS[@]}" -r "$ROOT/requirements.txt"
if ! "$VPY" -m pip check; then
  echo "[WARNING] pip check found dependency conflicts; review the output above."
fi

# Install or repair Ollama binaries without launching the server.
# Manual archive extraction avoids installer-managed service startup.
printf '\n'
read -r -p 'Install/repair Ollama locally (without starting it)? [y/N] ' configure
if [[ "$configure" =~ ^[Yy]$ ]]; then
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64) OLLAMA_ARCH=amd64 ;;
    aarch64) OLLAMA_ARCH=arm64 ;;
    *) echo "[ERROR] Unsupported architecture: $ARCH" >&2; exit 1 ;;
  esac
  "${APT[@]}" install -y tar zstd
  echo '[INFO] Checking Ollama installation...'
  REPAIR=0
  if ! command -v ollama >/dev/null 2>&1; then
    REPAIR=1
  elif [[ ! -x /usr/local/lib/ollama/llama-server ]]; then
    # This exact missing executable caused the user's observed 500 error.
    # Some newer builds may use a different layout; user can decline repair.
    echo '[WARNING] Expected llama-server is missing; this system previously failed inference.'
    REPAIR=1
  fi
  if (( REPAIR )); then
    read -r -p 'Download and reinstall the complete official Ollama archive? [y/N] ' confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo '[ERROR] Ollama is running. Stop it using ./unrun_ollama.sh, then rerun init.sh.' >&2
        exit 1
      fi
      ARCHIVE="$(mktemp /tmp/ollama-linux-XXXXXX.tar.zst)"
      trap 'rm -f "$ARCHIVE"' EXIT
      curl -fL --retry 3 "https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tar.zst" -o "$ARCHIVE"
      echo '[INFO] Extracting Ollama archive to /usr/local ...'
      if (( EUID == 0 )); then
        tar --zstd -xf "$ARCHIVE" -C /usr/local
      else
        sudo tar --zstd -xf "$ARCHIVE" -C /usr/local
      fi
      rm -f "$ARCHIVE"
      trap - EXIT
    fi
  else
    echo '[INFO] Ollama binary is already present; no reinstall needed.'
  fi
  if command -v ollama >/dev/null 2>&1; then
    ollama --version || true
    echo '[INFO] Ollama was NOT started by this script.'
    echo '[NEXT] Run ./run_ollama.sh --local or --all-ips to start and test it.'
  fi
fi
echo '[SUCCESS] INIT finished. No application or server was launched.'
echo "Python for GUI: $VPY"
echo '[IMPORTANT] The installer cannot change the PATH of the terminal that launched it.'
echo '[ACTION] For THIS terminal, run:'
echo '    export PATH="$HOME/.local/npm/bin:$PATH"'
echo '[ACTION] Or open a NEW terminal, then run: codex --version'
echo "[FALLBACK] Direct launcher: $NPM_PREFIX/bin/codex --version"

