#!/usr/bin/env bash
# Ubuntu bootstrap: Python, Node.js LTS, npm, Codex CLI, and project requirements.
# Run: bash init_updated.sh  (from config/ or anywhere)
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
  if [[ -d "$PROJECT_DIR/python" ]]; then
    echo '[INFO] Installing pipreqs and generating requirements.txt from python/ ...'
    "$PROJECT_DIR/.venv/bin/python" -m pip install pipreqs
    "$PROJECT_DIR/.venv/bin/python" -m pipreqs.pipreqs "$PROJECT_DIR/python" \
      --savepath "$PROJECT_DIR/requirements.txt" --force --mode no-pin --ignore .venv,venv
  else
    echo '[WARNING] python/ source folder not found; cannot generate requirements.txt.'
  fi
fi

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
  echo '[INFO] Installing project requirements...'
  "$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
else
  echo '[INFO] No requirements.txt at project root; skipping Python package installation.'
fi

echo '=== Initialization complete ==='
echo "Run: cd \"$PROJECT_DIR\""
echo 'In this terminal: source ~/.bashrc'
echo 'Verify: codex --version'
echo 'Start: codex --no-daemon  (if your Codex version supports this flag)'
echo 'Python environment: .venv/bin/python'
