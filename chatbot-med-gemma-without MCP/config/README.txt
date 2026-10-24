MedGemma Windows and Ubuntu setup

Place all scripts in project/config. Run init.bat on Windows or bash init.sh on Ubuntu.
Both installers ask whether to install Ollama and separately ask before downloading MedGemma 1.5 and Qwen3:8B on the CURRENT machine only. No SSH or remote installation.
Ubuntu Ollama setup is integrated into init.sh; no second script is needed.
Run uninit.bat or bash uninit.sh to remove Codex and the project .venv after confirmation. Uninit preserves Ollama and models.

If requirements.txt is missing, init scans project/python/ recursively for .py files.
If the directory or Python source files are missing, installation stops with an
explicit [ERROR] and tells you where to place the source or requirements.txt.
An existing requirements.txt is preserved and used without scanning Python files.
