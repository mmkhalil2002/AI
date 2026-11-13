Night Guard initialization files

Copy config/init.sh, config/init.bat, and requirements.txt into your project.
Keep the Python programs in python/ and Markdown files in md/.

Ubuntu 24.04: bash config/init.sh
Windows 11:   config\init.bat

Ubuntu script installs camera utilities and Python packages; no virtualenv.
Windows script preserves the previously supplied Node.js/Codex/Ollama setup,
including model downloads, so it can be slow and require substantial disk space.
Both create python/server.env and python/client.env only if missing.
Neither launches Night Guard.

Note: server/client Python programs are NOT included in this ZIP.
