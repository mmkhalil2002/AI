MedGemma — No-Virtual-Environment INIT/UNINIT

Copy the four scripts into PROJECT_ROOT/config/
Python source belongs in PROJECT_ROOT/python/
requirements.txt belongs in PROJECT_ROOT/ (generated if missing)

Windows:
  cd /d "C:\Users\Public\mkhalil\AI\AI\chatbot-med-gemma-without MCP\config"
  init.bat
  uninit.bat

Ubuntu:
  cd /path/to/project/config
  bash init.sh
  bash uninit.sh

INIT:
- Does not create a .venv; automatically removes existing PROJECT_ROOT/.venv.
- Installs/detects Python, pip, Node.js/npm and Codex.
- Generates requirements.txt if absent, after installing pipreqs.
- Prints errors if Python source files are missing.
- Installs Python packages into the current user's Python site-packages.
- Ubuntu uses pip --user --break-system-packages for externally managed Python.
  This is a deliberate tradeoff and may conflict with other Python programs.
- Optionally installs Ollama locally and downloads MedGemma 1.5 / Qwen3:8B.
  Ollama/models can consume multiple GB if you choose Yes.
- Does NOT start Ollama server, Codex, or the GUI.

UNINIT:
- Offers to uninstall global Codex and delete any remaining project .venv.
- Does NOT uninstall user Python packages (shared with other projects),
  Ollama, models, Python, Node.js, or delete project source/patient data.
- The init removes .venv automatically, without a separate confirmation.
