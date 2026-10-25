#!/usr/bin/env bash
# Run from config/ or anywhere. Does NOT launch Codex, Ollama server or Python GUI.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "=== MedGemma Ubuntu INIT ==="
echo "Project: $ROOT"
trap 'echo "[ERROR] Initialization failed at line $LINENO." >&2' ERR
command -v apt-get >/dev/null || { echo '[ERROR] Ubuntu/Debian apt-get required.' >&2; exit 1; }
sudo apt-get update
sudo apt-get install -y python3 python3-pip curl ca-certificates
node_major=0
if command -v node >/dev/null 2>&1; then
  node_major="$(node -p 'process.versions.node.split(".")[0]')"
fi
if ! command -v npm >/dev/null 2>&1 || (( node_major < 20 )); then
  echo '[INFO] Installing Node.js 22 from NodeSource...'
  sudo apt-get install -y gnupg
  curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/medgemma_nodesource_setup.sh
  sudo -E bash /tmp/medgemma_nodesource_setup.sh
  sudo apt-get install -y nodejs
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

# Never create a virtual environment. Delete an existing project .venv.
# Do not follow symlinks outside the project.
if [[ -L "$ROOT/.venv" ]]; then
  echo '[ERROR] .venv is a symbolic link. Remove it manually for safety.' >&2
  exit 1
fi
if [[ -d "$ROOT/.venv" ]]; then
  echo '[INFO] Removing existing project .venv...'
  rm -rf -- "$ROOT/.venv"
fi

# Modern Ubuntu marks system Python as externally managed (PEP 668).
# This explicit override installs into user site packages without creating a venv.
# It can still conflict with system/user Python packages; review requirements.txt.
PYTHON=python3
PIP_ARGS=(--user --break-system-packages)
if [[ ! -f "$ROOT/requirements.txt" ]]; then
  if [[ ! -d "$ROOT/python" ]]; then
    echo "[ERROR] Cannot generate requirements.txt: missing $ROOT/python" >&2
    echo '[ACTION] Add Python source files to python/ or create requirements.txt at project root.' >&2
    exit 1
  fi
  if ! find "$ROOT/python" -type f -name '*.py' -print -quit | grep -q .; then
    echo "[ERROR] Cannot generate requirements.txt: no .py files under $ROOT/python" >&2
    exit 1
  fi
  echo '[INFO] Installing pipreqs in user Python site-packages (no .venv)...'
  "$PYTHON" -m pip install "${PIP_ARGS[@]}" pipreqs
  echo '[INFO] Generating requirements.txt...'
  "$PYTHON" -m pipreqs.pipreqs "$ROOT/python" \
    --savepath "$ROOT/requirements.txt" --mode no-pin --use-local
  [[ -f "$ROOT/requirements.txt" ]] || { echo '[ERROR] requirements.txt was not created.' >&2; exit 1; }
fi
echo '[INFO] Installing application dependencies in user Python site-packages...'
"$PYTHON" -m pip install "${PIP_ARGS[@]}" -r "$ROOT/requirements.txt"
"$PYTHON" -m pip check

# Local Ollama installation is optional. Service startup is separate.
printf '\n'
read -r -p 'Configure Ollama and AI models on THIS computer? [y/N] ' configure
if [[ "$configure" =~ ^[Yy]$ ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    read -r -p 'Install Ollama locally? [y/N] ' install_ollama
    if [[ "$install_ollama" =~ ^[Yy]$ ]]; then
      echo '[INFO] Downloading official Ollama installer...'
      curl -fsSL https://ollama.com/install.sh -o /tmp/medgemma_ollama_install.sh
      echo '[INFO] Review /tmp/medgemma_ollama_install.sh if desired before running.'
      sh /tmp/medgemma_ollama_install.sh
    fi
  fi
  if command -v ollama >/dev/null 2>&1; then
    ollama --version || true
    if ollama list >/dev/null 2>&1; then
      read -r -p 'Download MedGemma 1.5? [y/N] ' med
      if [[ "$med" =~ ^[Yy]$ ]]; then
        ollama pull medgemma1.5 || echo '[WARNING] MedGemma download failed.'
      fi
      read -r -p 'Download Qwen3:8B? [y/N] ' qwen
      if [[ "$qwen" =~ ^[Yy]$ ]]; then
        ollama pull qwen3:8b || echo '[WARNING] Qwen3 download failed.'
      fi
    else
      echo '[ACTION] Start the Ollama service separately and rerun init.sh to download models.'
    fi
  else
    echo '[INFO] Ollama not installed; skipping models.'
  fi
fi
echo '[SUCCESS] INIT finished. No application or server was launched.'
echo "Python for GUI: $(command -v python3)"
echo 'Open a new terminal for the updated Codex PATH.'
