@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Run from config\. Does NOT start Codex, Ollama, server or GUI.
rem Night Guard setup: dependencies plus existing Node/Codex/Ollama functionality.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VENV=%ROOT%\.venv"
set "PYTHON_DIR=%ROOT%\python"
echo === Night Guard Windows INIT ===
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
if not exist "%ROOT%\requirements.txt" (
  if not exist "%ROOT%\python\" (
    echo [ERROR] Cannot generate requirements.txt: "%ROOT%\python" does not exist.
    echo [ACTION] Put .py files in python\ or create requirements.txt in project root.
    goto failed
  )
  dir /s /b /a-d "%ROOT%\python\*.py" 2>nul | findstr /r "." >nul
  if errorlevel 1 (
    echo [ERROR] Cannot generate requirements.txt: no .py files found under "%ROOT%\python".
    goto failed
  )
  echo [INFO] Installing pipreqs into user Python packages...
  "%VPY%" -m pip install --user pipreqs
  if errorlevel 1 goto failed
  echo [INFO] Generating requirements.txt...
  "%VPY%" -m pipreqs.pipreqs "%ROOT%\python" --savepath "%ROOT%\requirements.txt" --mode no-pin --use-local
  if errorlevel 1 goto failed
  if not exist "%ROOT%\requirements.txt" goto failed
)
echo [INFO] Installing Python dependencies into user Python packages...
"%VPY%" -m pip install --user -r "%ROOT%\requirements.txt"
if errorlevel 1 goto failed
"%VPY%" -m pip check
if errorlevel 1 goto failed

rem Create separate Night Guard configuration files only if absent.
if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
if not exist "%PYTHON_DIR%\server.env" (
  >"%PYTHON_DIR%\server.env" echo CAMERA_COUNT=4
  >>"%PYTHON_DIR%\server.env" echo FRONT_CAMERA=0
  >>"%PYTHON_DIR%\server.env" echo BACK_CAMERA=1
  >>"%PYTHON_DIR%\server.env" echo LEFT_CAMERA=2
  >>"%PYTHON_DIR%\server.env" echo RIGHT_CAMERA=3
  >>"%PYTHON_DIR%\server.env" echo WIDTH=640
  >>"%PYTHON_DIR%\server.env" echo HEIGHT=480
  >>"%PYTHON_DIR%\server.env" echo FPS=15
  >>"%PYTHON_DIR%\server.env" echo JPEG_QUALITY=80
  >>"%PYTHON_DIR%\server.env" echo SERVER_HOST=127.0.0.1
  >>"%PYTHON_DIR%\server.env" echo SERVER_PORT=14095
  >>"%PYTHON_DIR%\server.env" echo CONTROL_PORT=14098
)
if not exist "%PYTHON_DIR%\client.env" (
  >"%PYTHON_DIR%\client.env" echo SERVER_URL=http://127.0.0.1:14095
  >>"%PYTHON_DIR%\client.env" echo GUI_TITLE=Night Guard
  >>"%PYTHON_DIR%\client.env" echo GUI_WIDTH=1400
  >>"%PYTHON_DIR%\client.env" echo GUI_HEIGHT=900
  >>"%PYTHON_DIR%\client.env" echo CONTROL_PANEL_HEIGHT=200
)
rem Ollama and both required models are mandatory for a successful INIT.
rem Only a temporary Ollama server is started if necessary.
echo [INFO] Ensuring Ollama and required AI models are installed...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
where ollama.exe >nul 2>&1
if errorlevel 1 (
  where winget.exe >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] winget unavailable. Install Ollama from https://ollama.com/download/windows
    goto failed
  )
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto failed
  set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
)
where ollama.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Ollama not found on PATH. Open a new CMD and rerun init.bat.
  goto failed
)
ollama --version
rem Use PowerShell to manage a temporary local-only server and check every model.
rem The PowerShell script stops only the process it starts, never an existing server.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $models=@('medgemma1.5:latest','qwen3:8b'); $server=$null; try {" ^
  "  try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null; $ready=$true } catch { $ready=$false };" ^
  "  if (-not $ready) { Write-Host '[INFO] Starting temporary Ollama server...'; $exe=(Get-Command ollama.exe).Source; $server=Start-Process -FilePath $exe -ArgumentList 'serve' -PassThru -WindowStyle Hidden; for($i=0;$i -lt 45;$i++){ Start-Sleep -Seconds 2; try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; $ready=$true; break } catch {} }; if(-not $ready){ throw 'Ollama API did not become ready' } };" ^
  "  foreach($model in $models){ Write-Host ('[INFO] Checking '+$model); $installed=@(ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] }); if($installed -notcontains $model){ Write-Host ('[INFO] Downloading '+$model); & ollama pull $model; if($LASTEXITCODE -ne 0){ throw ('Download failed: '+$model) } }; $installed=@(ollama list | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] }); if($installed -notcontains $model){ throw ('Verification failed: '+$model) }; Write-Host ('[OK] Verified: '+$model) };" ^
  "} catch { Write-Host ('[ERROR] '+$_.Exception.Message); exit 1 } finally { if($null -ne $server -and -not $server.HasExited){ Write-Host '[INFO] Stopping temporary Ollama server...'; Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue } }"
if errorlevel 1 goto failed
:done
echo.
echo [SUCCESS] INIT finished. No servers or GUI were started.
echo Python for Night Guard: "%PYEXE%"
echo Server config: "%PYTHON_DIR%\server.env"
echo Client config: "%PYTHON_DIR%\client.env"
echo Codex: "%NPM_PREFIX%\codex.cmd"
echo.
echo [IMPORTANT] This batch cannot change PATH in the CMD window that called it.
echo [INFO] User PATH has been configured for future CMD terminals.
echo [ACTION] Existing CMD windows cannot inherit PATH changes automatically.
echo [ACTION] In this CURRENT CMD window, run:
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
