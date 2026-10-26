# MedGemma initialization: Windows and Ubuntu

Copy `init.bat` and `init.sh` into the project `config/` directory. These scripts use the project parent directory for `python/` and `requirements.txt`. Neither starts the GUI or Ollama service.

## Windows
Run `config\init.bat` from Command Prompt. It installs/updates Node.js, Codex CLI and Python dependencies, and optionally Ollama/models. It attempts to persist the npm user prefix in User PATH. **An already-open CMD cannot inherit PATH changes made inside a batch with `setlocal`.** After setup, close CMD and open a new CMD, then run `codex --version`. For the existing CMD session run `set "PATH=%APPDATA%\npm;%PATH%"` and then `codex --version`. The full launcher path is `%APPDATA%\npm\codex.cmd`.

## Ubuntu
Run `bash config/init.sh`. It configures `$HOME/.local/npm/bin` in `.bashrc`, installs Codex, and installs Python packages in the user site without a virtual environment. For the current shell run `export PATH="$HOME/.local/npm/bin:$PATH"`; new Bash terminals should pick up the PATH. Ubuntu 24.04+ may require pip's `--break-system-packages` flag even for `--user` installs; the script uses it, so review the impact on your user Python environment.

## Notes
- Both scripts may install software and download packages; inspect before executing.
- Neither script guarantees the old parent shell is modified; reopen the terminal or use the one-line PATH fix.
- These scripts are based on the supplied Windows `init(4).bat` and the existing Ubuntu `init.sh`, with PATH guidance and no-venv consistency improvements.
