MedGemma INIT and UNINIT — Windows and Ubuntu

Put all four scripts in the project config/ directory.
Windows: run init.bat or uninit.bat from CMD.
Ubuntu: bash init.sh or bash uninit.sh.

INIT: Does not create/use .venv. Deletes existing project-root .venv. Installs Python dependencies into existing user Python environment, installs Codex, optionally installs Ollama/models. Does not start GUI or server.

UNINIT: Preserves Python installation, pip, all installed Python packages (including pipreqs), all Python-related environment variables and PATH settings. Preserves Node.js, npm, Ollama, models, requirements.txt, .env, source and patient data. Does not stop running servers. Asks before optionally uninstalling globally installed Codex and deleting a leftover project .venv.

Note: Ubuntu init uses pip --user --break-system-packages, which can affect Python package compatibility. Ollama model downloads can be large if approved.
