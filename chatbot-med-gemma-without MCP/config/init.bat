@echo off
setlocal EnableExtensions
REM Bootstrap Windows development tools for the simplified Medical AI project.
REM Does not install Ollama models or change .env. Run from CMD.
cd /d "%~dp0"
echo === Medical AI Windows initialization ===
echo Directory: %CD%
echo.

REM Add common Node.js and npm paths to this CMD session, even if PATH is stale.
if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
if exist "%APPDATA%\npm" set "PATH=%APPDATA%\npm;%PATH%"
if exist "%LOCALAPPDATA%\Programs\nodejs\node.exe" set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"

where node.exe >nul 2>&1
if errorlevel 1 (
  echo [1/4] Node.js missing. Installing Node.js LTS...
  where winget.exe >nul 2>&1
  if errorlevel 1 goto :manual_node
  winget install --id OpenJS.NodeJS.LTS --exact --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :manual_node
  if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
  where node.exe >nul 2>&1
  if errorlevel 1 (
    echo [ACTION] Node.js installation completed but PATH is not refreshed.
    echo Close CMD, open a new CMD, and rerun init.bat.
    exit /b 0
  )
)
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [ERROR] node exists but npm.cmd is missing. Repair Node.js LTS from https://nodejs.org/en/download
  exit /b 1
)
echo [1/4] Node.js and npm:
call node --version
call npm.cmd --version
if errorlevel 1 goto :failed

REM Locate Python or install it. Do not rely on the Windows Store python alias.
set "PYTHON_CMD="
where py.exe >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys; print(sys.version)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
  where python.exe >nul 2>&1
  if not errorlevel 1 (
    python -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
  )
)
if not defined PYTHON_CMD (
  echo [2/4] Python missing. Installing Python 3...
  where winget.exe >nul 2>&1
  if errorlevel 1 goto :manual_python
  winget install --id Python.Python.3.13 --exact --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :manual_python
  echo [ACTION] Close CMD, open a new CMD, and rerun init.bat to detect Python.
  exit /b 0
)
echo [2/4] Python:
%PYTHON_CMD% --version
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 %PYTHON_CMD% -m ensurepip --upgrade
%PYTHON_CMD% -m pip --version
if errorlevel 1 goto :failed

REM Install or update Codex and add npm's global bin prefix to this session.
echo [3/4] Installing/updating OpenAI Codex CLI...
call npm.cmd install --global @openai/codex
if errorlevel 1 goto :failed
for /f "usebackq delims=" %%P in (`npm.cmd config get prefix 2^>nul`) do set "NPM_PREFIX=%%P"
if defined NPM_PREFIX set "PATH=%NPM_PREFIX%;%PATH%"
if exist "%APPDATA%\npm" set "PATH=%APPDATA%\npm;%PATH%"
where codex.cmd >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Codex installed but its executable is not on PATH.
  echo npm global prefix: %NPM_PREFIX%
  echo Add that directory to your user PATH, then open a new CMD.
  echo You can try: npx --yes @openai/codex
  exit /b 1
)
call codex.cmd --version
if errorlevel 1 goto :failed

REM Install dependencies only when Codex has generated requirements.txt.
echo [4/4] Python project packages...
if exist "requirements.txt" (
  %PYTHON_CMD% -m pip install -r "requirements.txt"
  if errorlevel 1 goto :failed
) else (
  echo [INFO] No requirements.txt yet. Rerun init.bat after Codex generates it.
)
echo.
echo [SUCCESS] Initialization complete.
echo To start Codex in this directory, run: codex
echo If this CMD session still cannot find Codex, open a NEW CMD window.
exit /b 0

:manual_node
echo [ERROR] Automatic Node.js installation unavailable or failed.
echo Install Node.js LTS from https://nodejs.org/en/download
 echo Then close CMD, reopen it, and rerun init.bat.
exit /b 1
:manual_python
echo [ERROR] Automatic Python installation unavailable or failed.
echo Install Python 3 from https://www.python.org/downloads/windows/
echo Then close CMD, reopen it, and rerun init.bat.
exit /b 1
:failed
echo [ERROR] Initialization failed. Review the message above.
exit /b 1
