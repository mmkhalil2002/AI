@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "HOST=127.0.0.1:11434"
set "PULL=0"
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--local" set "HOST=127.0.0.1:11434"& shift & goto parse
if /I "%~1"=="--all-ips" set "HOST=0.0.0.0:11434"& shift & goto parse
if /I "%~1"=="--pull-models" set "PULL=1"& shift & goto parse
echo Usage: run_ollama.bat [--local^|--all-ips] [--pull-models]
exit /b 2
:parsed
where ollama >nul 2>&1
if errorlevel 1 (echo [ERROR] Ollama not found on PATH.& exit /b 1)
curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 (
  echo [WARNING] Ollama is already running. This script will not change its existing listening address.
  echo [ACTION] Quit Ollama from the system tray and stop any existing ollama serve process, then rerun.
  exit /b 1
)
echo [INFO] Starting Ollama with OLLAMA_HOST=%HOST%
start "Ollama Server" /min cmd /c "set OLLAMA_HOST=%HOST%&& ollama serve"
set /a attempts=0
:wait_loop
timeout /t 1 /nobreak >nul
curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
  set /a attempts+=1
  if !attempts! GEQ 30 (echo [ERROR] Ollama did not become ready.& exit /b 1)
  goto wait_loop
)
echo [SUCCESS] Ollama API ready. Requested bind: %HOST%
ollama list
if "%PULL%"=="1" (
  ollama pull medgemma1.5 || exit /b 1
  ollama pull qwen3:8b || exit /b 1
)
if "%HOST%"=="0.0.0.0:11434" echo [SECURITY] Restrict TCP 11434 to trusted LAN hosts in Windows Firewall.
endlocal
