#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo '=== MedGemma Ubuntu UNINIT ==='
echo 'Optionally remove global Codex CLI and project .venv.'
echo 'Preserves Python, Node.js, npm, PATH, Ollama, models, requirements.txt, code and data.'
echo 'Does NOT stop any running server.'
read -r -p 'Proceed with cleanup? [y/N] ' confirm
[[ "$confirm" =~ ^[Yy]$ ]] || exit 0
read -r -p 'Remove globally installed Codex CLI (affects other projects)? [y/N] ' codex_confirm
if [[ "$codex_confirm" =~ ^[Yy]$ ]]; then
  if command -v npm >/dev/null 2>&1; then
    npm uninstall -g @openai/codex || echo '[WARNING] Codex removal failed.'
  else
    echo '[WARNING] npm not found; Codex removal skipped.'
  fi
fi
if [[ -d "$ROOT/.venv" ]]; then
  read -r -p 'Delete project .venv directory? [y/N] ' venv_confirm
  if [[ "$venv_confirm" =~ ^[Yy]$ ]]; then
    rm -rf -- "$ROOT/.venv"
  fi
fi
echo '[SUCCESS] Cleanup finished. Application and Ollama files preserved.'
