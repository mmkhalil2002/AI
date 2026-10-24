#!/usr/bin/env bash
# Ubuntu bootstrap: Python, Node.js LTS, npm, Codex CLI, and project requirements.
# Run: bash init.sh  (from config/ or anywhere)
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "config" ]]; then
  PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
else
  PROJECT_DIR="$SCRIPT_DIR"
fi
printf '=== Medical AI - Ubuntu initialization ===\nProject: %s\n' "$PROJECT_DIR"

if ! command -v apt-get >/dev/null 2>&1; then
  echo '[ERROR] This installer requires Ubuntu/Debian apt-get.' >&2
  exit 1
fi

# Install Python and required OS tools if missing.
if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip curl ca-certificates
fi

# Modern Node.js is required for Codex. Ubuntu's default Node may be too old.
NODE_MAJOR=0
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
fi
if ! command -v npm >/dev/null 2>&1 || (( NODE_MAJOR < 20 )); then
  echo '[INFO] Installing modern Node.js and npm (NodeSource Node 22.x).'
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
printf 'Node: %s | npm: %s\n' "$(node --version)" "$(npm --version)"

# npm global packages are installed in a writable, user-owned location.
NPM_PREFIX="$HOME/.local/npm"
mkdir -p "$NPM_PREFIX/bin" "$HOME/.npm"
npm config set prefix "$NPM_PREFIX" --location=user
export PATH="$NPM_PREFIX/bin:$PATH"

# Persist the npm executable directory for NEW interactive bash terminals.
BASHRC="$HOME/.bashrc"
PATH_LINE='export PATH="$HOME/.local/npm/bin:$PATH"'
touch "$BASHRC"
if ! grep -Fqx "$PATH_LINE" "$BASHRC"; then
  printf '\n# User-installed npm commands (including Codex)\n%s\n' "$PATH_LINE" >> "$BASHRC"
fi

# Install Codex or upgrade the existing package in the configured prefix.
echo '[INFO] Installing/updating OpenAI Codex CLI...'
npm install --global @openai/codex
if ! command -v codex >/dev/null 2>&1; then
  echo "[ERROR] Codex not found on PATH. Expected: $NPM_PREFIX/bin/codex" >&2
  exit 1
fi
echo "Codex: $(codex --version)"

# Isolate Python dependencies from the system interpreter.
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo '[INFO] Creating project Python environment...'
  python3 -m venv "$PROJECT_DIR/.venv"
fi
# Bootstrap dependency scanner only when requirements.txt is missing.
# Never overwrite a hand-maintained requirements.txt.
if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
  if [[ ! -d "$PROJECT_DIR/python" ]]; then
    echo "[ERROR] Python source directory not found: $PROJECT_DIR/python" >&2
    echo '[ERROR] Cannot generate requirements.txt. Add your Python files to python/,' >&2
    echo "[ERROR] or create requirements.txt manually in $PROJECT_DIR" >&2
    exit 1
  fi
  if ! find "$PROJECT_DIR/python" -type f -name '*.py' -print -quit | grep -q .; then
    echo "[ERROR] No .py source files found under $PROJECT_DIR/python" >&2
    echo '[ERROR] Cannot generate requirements.txt. Add your Python program first,' >&2
    echo "[ERROR] or create requirements.txt manually in $PROJECT_DIR" >&2
    exit 1
  fi
  echo '[INFO] Installing pipreqs and generating requirements.txt from python/ ...'
  "$PROJECT_DIR/.venv/bin/python" -m pip install pipreqs
  if ! "$PROJECT_DIR/.venv/bin/python" -m pipreqs.pipreqs "$PROJECT_DIR/python" \
    --savepath "$PROJECT_DIR/requirements.txt" --force --mode no-pin --ignore .venv,venv; then
    echo '[ERROR] Dependency detection failed. Check Python sources or create requirements.txt manually.' >&2
    exit 1
  fi
  if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
    echo '[ERROR] pipreqs completed but requirements.txt was not created.' >&2
    exit 1
  fi
fi

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
  echo '[INFO] Installing project requirements...'
  "$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
else
  echo '[INFO] No requirements.txt at project root; skipping Python package installation.'
fi

# Optional Ollama installation on THIS Ubuntu computer only.
# Run this same init.sh on another computer if you want models installed there.
printf '\nOptional local Ollama and AI models setup\n'
read -r -p 'Configure Ollama and AI models on THIS Ubuntu computer? [y/N] ' models_choice
if [[ "$models_choice" =~ ^[Yy]$ ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    read -r -p 'Install Ollama locally? [y/N] ' ollama_choice
    if [[ "$ollama_choice" =~ ^[Yy]$ ]]; then
      if ! command -v curl >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y curl ca-certificates
      fi
      # Official Ollama Linux installation script; review source before approving.
      curl -fsSL https://ollama.com/install.sh | sh
    fi
  else
    echo '[INFO] Ollama is already installed.'
  fi

  if command -v ollama >/dev/null 2>&1; then
    if ! ollama list >/dev/null 2>&1; then
      echo '[INFO] Ollama service is not ready; attempting to start it...'
      if command -v systemctl >/dev/null 2>&1; then
        sudo systemctl start ollama || true
      fi
    fi
    if ollama list >/dev/null 2>&1; then
      read -r -p 'Download MedGemma 1.5 on THIS computer? [y/N] ' med_choice
      if [[ "$med_choice" =~ ^[Yy]$ ]]; then
        ollama pull medgemma1.5 || echo '[WARNING] MedGemma download failed.'
      fi
      read -r -p 'Download Qwen3:8B on THIS computer? [y/N] ' qwen_choice
      if [[ "$qwen_choice" =~ ^[Yy]$ ]]; then
        ollama pull qwen3:8b || echo '[WARNING] Qwen3 download failed.'
      fi
    else
      echo '[WARNING] Ollama service unavailable. Start Ollama and rerun init.sh to download models.'
    fi
  else
    echo '[INFO] Ollama not installed; model downloads skipped.'
  fi
fi

echo '=== Initialization complete ==='
echo "Run: cd \"$PROJECT_DIR\""
echo 'In this terminal: source ~/.bashrc'
echo 'Verify: codex --version'
echo 'Start: codex --no-daemon  (if your Codex version supports this flag)'
echo 'Python environment: .venv/bin/python'
