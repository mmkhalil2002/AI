@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
echo === MedGemma Windows UNINIT ===
echo This can remove Codex CLI globally and this project's .venv.
echo It PRESERVES Node.js, Python, npm, pipreqs outside .venv, Ollama,
echo downloaded AI models, PATH, requirements.txt, code, and patient data.
echo It does NOT stop running servers.
choice /C YN /N /M "Proceed with cleanup? [Y/N] "
if errorlevel 2 exit /b 0
choice /C YN /N /M "Remove globally installed Codex CLI (affects other projects)? [Y/N] "
if errorlevel 2 goto venv
if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
set "PATH=%APPDATA%\npm;%PATH%"
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [WARNING] npm unavailable. Cannot remove Codex.
) else (
  call npm.cmd uninstall -g @openai/codex
  if errorlevel 1 echo [WARNING] Codex removal failed.
)
:venv
if exist "%ROOT%\.venv\" (
  choice /C YN /N /M "Delete project .venv directory? [Y/N] "
  if errorlevel 2 goto done
  rmdir /s /q "%ROOT%\.venv"
  if exist "%ROOT%\.venv\" echo [WARNING] Could not delete .venv completely.
)
:done
echo [SUCCESS] Cleanup finished. Source code, requirements.txt, Ollama and models preserved.
exit /b 0
