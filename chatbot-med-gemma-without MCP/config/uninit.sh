#!/usr/bin/env bash
# MedGemma project cleanup for Ubuntu.
# Removes global Codex CLI from the active npm prefix and project-local .venv.
# Preserves Python, Node.js, npm, PATH, pipreqs, requirements.txt, code and data.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
printf '=== MedGemma Ubuntu uninitialization ===\nProject: %s\n\n' "$PROJECT_ROOT"
printf '%s\n' 'This removes the globally installed OpenAI Codex CLI and project .venv.' \
  'It does NOT remove Python, Node.js, npm, pipreqs, PATH settings,' \
  'requirements.txt, code, patient records, or Ollama models.'
read -r -p 'Continue? [y/N]: ' confirm
case "$confirm" in y|Y|yes|YES) ;; *) echo 'Cancelled. No changes made.'; exit 0 ;; esac
if command -v npm >/dev/null 2>&1; then
  echo '[1/2] Removing Codex CLI from active npm global prefix...'
  if ! npm uninstall -g @openai/codex; then
    echo '[WARNING] Codex removal failed; check npm permissions/prefix.' >&2
  fi
else
  echo '[1/2] npm not found; skipping Codex uninstall.'
fi
if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  echo '[2/2] Removing project virtual environment...'
  rm -rf -- "$PROJECT_ROOT/.venv"
else
  echo '[2/2] No project .venv found.'
fi
echo 'Finished. requirements.txt and all project files were preserved.'
