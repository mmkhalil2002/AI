#!/usr/bin/env bash
# Medical AI development bootstrap for Ubuntu.
# Installs Python tooling, Node.js LTS, Codex CLI and requirements.txt packages.
# Uses a project-local Python virtual environment to avoid Ubuntu's externally-managed
# system Python restrictions. Does not install remote Ollama models or modify .env.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
echo '=== Medical AI - Ubuntu initialization ==='
echo "Project: $PWD"

if ! command -v apt-get >/dev/null 2>&1; then
  echo '[ERROR] This script targets Ubuntu with apt-get.' >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  echo 'Installing Python 3, venv, pip, curl, and certificates...'
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip curl ca-certificates
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo 'Installing Node.js and npm from Ubuntu repositories...'
  sudo apt-get update
  sudo apt-get install -y nodejs npm
fi

# Codex requires a modern Node.js version. Do not silently continue with an old distro package.
node_major="$(node -p 'process.versions.node.split(".")[0]')"
if (( node_major < 18 )); then
  echo "[ERROR] Node.js $(node --version) is too old. Install current Node.js LTS and rerun." >&2
  echo 'See https://nodejs.org/en/download' >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo 'Installing OpenAI Codex CLI...'
  if ! npm install --global @openai/codex; then
    echo '[ERROR] npm global install failed. Configure an npm user prefix (or use a Node version manager), then rerun.' >&2
    exit 1
  fi
else
  echo "Codex CLI already installed: $(codex --version)"
fi

if [[ ! -x .venv/bin/python ]]; then
  echo 'Creating local Python environment (.venv)...'
  if ! python3 -m venv .venv; then
    echo '[ERROR] Python venv creation failed. Try: sudo apt-get install python3-venv' >&2
    exit 1
  fi
fi

if [[ -f requirements.txt ]]; then
  echo 'Installing Python packages from requirements.txt...'
  .venv/bin/python -m pip install -r requirements.txt
else
  echo '[INFO] requirements.txt not found; skipping project Python packages.'
  echo '[INFO] Rerun bash init.sh after Codex generates requirements.txt.'
fi

echo 'Initialization complete.'
echo 'Verify: codex --version'
echo 'Start Codex: codex'
echo 'After app is generated: .venv/bin/python client_gui.py'
