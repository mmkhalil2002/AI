@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Run from config\. Does NOT start Codex, Ollama, or the Python GUI.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VENV=%ROOT%\.venv"
echo === MedGemma Windows INIT ===
echo Project: "%ROOT%"

rem Node.js and npm
if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
if exist "%LOCALAPPDATA%\Programs\nodejs\node.exe" set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"
where node.exe >nul 2>&1
if errorlevel 1 (
  where winget.exe >nul 2>&1
  if errorlevel 1 goto node_error
  echo [INFO] Installing Node.js LTS...
  winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto node_error
  if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
)
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [ACTION] Reopen CMD and rerun init.bat after Node.js installation.
  exit /b 1
)
node --version
call npm.cmd --version
if errorlevel 1 goto failed

rem Use the actual current account, never a hardcoded user profile.
if not defined APPDATA goto failed
set "NPM_PREFIX=%APPDATA%\npm"
if not exist "%NPM_PREFIX%" mkdir "%NPM_PREFIX%"
if not exist "%NPM_PREFIX%" goto failed
call npm.cmd config set prefix "%NPM_PREFIX%"
if errorlevel 1 goto failed
set "PATH=%NPM_PREFIX%;%PATH%"
for /f "delims=" %%I in ('where node.exe') do if not defined NODE_DIR for %%J in ("%%I") do set "NODE_DIR=%%~dpJ"
if defined NODE_DIR (
  powershell.exe -NoProfile -Command "$add=@($env:NODE_DIR,$env:NPM_PREFIX);$old=[Environment]::GetEnvironmentVariable('Path','User');$a=@($old -split ';' | Where-Object { $_ });foreach($p in $add){$p=$p.TrimEnd('\');if($a -notcontains $p){$a+=$p}};[Environment]::SetEnvironmentVariable('Path',($a -join ';'),'User')"
  if errorlevel 1 (
    echo [WARNING] Could not save user PATH permanently.
    echo [ACTION] Add "%NPM_PREFIX%" to User Path manually.
  )
)

rem Find a working Python, not the Windows Store alias.
set "PYEXE="
where py.exe >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%I in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do if not defined PYEXE set "PYEXE=%%I"
)
if not defined PYEXE (
  where python.exe >nul 2>&1
  if not errorlevel 1 (
    for /f "delims=" %%I in ('python -c "import sys;print(sys.executable)" 2^>nul') do if not defined PYEXE set "PYEXE=%%I"
  )
)
if not defined PYEXE (
  for /d %%I in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%~fI\python.exe" set "PYEXE=%%~fI\python.exe"
)
if not defined PYEXE (
  where winget.exe >nul 2>&1
  if errorlevel 1 goto python_error
  echo [INFO] Installing Python 3.13...
  winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
  echo [ACTION] Open a new CMD and rerun init.bat to finish setup.
  exit /b 1
)
"%PYEXE%" --version
if errorlevel 1 goto python_error
"%PYEXE%" -m pip --version >nul 2>&1
if errorlevel 1 "%PYEXE%" -m ensurepip --upgrade
if errorlevel 1 goto failed

rem Codex CLI, installed under current user npm prefix.
echo [INFO] Installing/updating Codex CLI...
call npm.cmd install -g @openai/codex
if errorlevel 1 goto failed
if not exist "%NPM_PREFIX%\codex.cmd" (
  echo [ERROR] Codex launcher missing at "%NPM_PREFIX%\codex.cmd".
  goto failed
)
call "%NPM_PREFIX%\codex.cmd" --version
if errorlevel 1 goto failed

rem Never create a virtual environment; delete only this project's existing .venv.
if exist "%ROOT%\.venv\" (
  echo [INFO] Removing existing project .venv directory...
  rmdir /s /q "%ROOT%\.venv"
  if exist "%ROOT%\.venv\" (
    echo [ERROR] Could not remove "%ROOT%\.venv". Close programs using it.
    goto failed
  )
)
rem Install pipreqs and application packages in the current user's Python environment.
set "VPY=%PYEXE%"
rem Generate or refresh requirements with explicit MCP compatibility constraints.
"%VPY%" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 goto python_error
"%VPY%" "%~dp0generate_requirements.py"
if errorlevel 1 goto failed
echo [INFO] Installing Python dependencies into user Python packages...
"%VPY%" -m pip install --user -r "%ROOT%\requirements.txt"
if errorlevel 1 goto failed
"%VPY%" -m pip check
if errorlevel 1 goto failed

rem Ollama is OPTIONAL and installed only on this machine.
rem Running the Ollama service is intentionally handled separately.
echo.
choice /C YN /N /M "Configure Ollama and AI models on THIS computer? [Y/N] "
if errorlevel 2 goto done
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
where ollama.exe >nul 2>&1
if errorlevel 1 (
  choice /C YN /N /M "Install Ollama locally? [Y/N] "
  if errorlevel 2 goto done
  where winget.exe >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] winget unavailable. Install Ollama manually from https://ollama.com/download/windows
    goto done
  )
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo [WARNING] Ollama installation failed.
    goto done
  )
  set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
)
where ollama.exe >nul 2>&1
if errorlevel 1 (
  echo [WARNING] Ollama not on PATH yet. Open a new CMD and rerun init.bat.
  goto done
)
ollama --version
echo [INFO] Ollama is installed. Model downloads require a running Ollama service.
ollama list >nul 2>&1
if errorlevel 1 (
  echo [ACTION] Start Ollama separately, then rerun init.bat to download models.
  goto done
)
choice /C YN /N /M "Download MedGemma 1.5? [Y/N] "
if errorlevel 2 goto done
ollama pull medgemma1.5
if errorlevel 1 echo [WARNING] MedGemma model download failed.
:done
echo.
echo [SUCCESS] INIT finished. No servers or GUI were started.
echo Python for GUI: "%PYEXE%"
echo Codex: "%NPM_PREFIX%\codex.cmd"
echo.
echo [IMPORTANT] This batch cannot change PATH in the CMD window that called it.
echo [ACTION] For this CURRENT CMD window, run:
echo     set "PATH=%NPM_PREFIX%;%%PATH%%"
echo [ACTION] Or CLOSE this CMD window and OPEN a NEW one.
echo [ACTION] Then run: codex --version
echo [FALLBACK] Always works using full path:
echo     "%NPM_PREFIX%\codex.cmd" --version
exit /b 0

:node_error
echo [ERROR] Node.js/npm unavailable. Install Node.js LTS and rerun.
exit /b 1
:python_error
echo [ERROR] Python unavailable. Install Python 3 and rerun.
exit /b 1
:failed
echo [ERROR] Initialization failed. See messages above.
exit /b 1
