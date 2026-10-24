MedGemma installation and cleanup — Windows / Ubuntu

Put all four scripts in PROJECT_ROOT/config/.
Put Python source files in PROJECT_ROOT/python/.
The generated requirements.txt belongs in PROJECT_ROOT/.

Windows:
  cd /d "C:\Users\Public\mkhalil\AI\AI\chatbot-med-gemma-without MCP\config"
  init.bat
  uninit.bat

Ubuntu:
  cd /path/to/project/config
  bash init.sh
  bash uninit.sh

INIT:
- Installs/detects Python, pip, Node.js/npm, Codex.
- Uses a project-local .venv for Python packages.
- If requirements.txt is missing, installs pipreqs into .venv and scans python/.
- Displays a clear ERROR if the Python source directory or .py files are missing.
- Installs packages listed in requirements.txt and runs pip check.
- Optionally installs Ollama locally and downloads MedGemma 1.5 / Qwen3:8B.
- Does NOT launch Codex, the GUI, or Ollama service. Start those separately.
- On Windows, Python/Node installations may require a new CMD window and rerun.
- pipreqs cannot detect every dynamic import; review requirements.txt.

UNINIT:
- Asks before removing globally installed Codex (affects other projects).
- Asks before deleting the project's .venv.
- Preserves Python, Node, npm, Ollama, models, PATH, requirements.txt and data.
- Does not uninstall or stop Ollama.

Ollama model downloads require Ollama service to be running already.
Use your own separate start/stop script for that service.
