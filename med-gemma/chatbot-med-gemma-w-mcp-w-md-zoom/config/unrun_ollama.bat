@echo off
setlocal EnableExtensions
rem ===========================================================================
rem unrun_ollama.bat - Stop the Ollama server on Windows 10/11.
rem
rem USAGE:
rem   unrun_ollama.bat          Stop Ollama without deleting models
rem   unrun_ollama.bat -help    Display help
rem
rem Stops the Ollama Windows process and, when possible, the Ollama service.
rem This affects Ollama running ON THIS WINDOWS PC, not on a remote Ubuntu PC.
rem It does not uninstall Ollama or delete downloaded models.
rem If Ollama auto-starts from the tray/startup, it may start again later.
rem ===========================================================================
if /I "%~1"=="-help" goto help
if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help
if not "%~1"=="" (
  echo [ERROR] Unknown option: %~1
  exit /b 1
)
where ollama.exe >nul 2>&1
if errorlevel 1 echo [NOTE] ollama.exe is not on PATH; checking running processes anyway.
rem Stop the service if one exists; stopping a protected service may need Admin.
sc query ollama >nul 2>&1
if not errorlevel 1 (
  echo [INFO] Requesting Ollama service stop...
  sc stop ollama >nul 2>&1
)
rem Ollama may run as a tray app or a console process. Stop both.
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if not errorlevel 1 (
  echo [INFO] Stopping Ollama processes...
  taskkill /F /T /IM ollama.exe >nul 2>&1
)
rem Some versions launch a separate Ollama application process.
tasklist /FI "IMAGENAME eq Ollama app.exe" 2>nul | find /I "Ollama app.exe" >nul
if not errorlevel 1 taskkill /F /T /IM "Ollama app.exe" >nul 2>&1
rem Check API on the usual local port; this is a warning, not a destructive fix.
curl.exe --silent --max-time 2 http://127.0.0.1:11434/ >nul 2>&1
if not errorlevel 1 (
  echo [WARNING] Ollama is still responding on port 11434.
  echo [NOTE] Check Windows Services or the Ollama tray app. Administrator rights may be required.
  exit /b 1
)
echo [SUCCESS] Ollama stopped or was already stopped. Models remain installed.
exit /b 0
:help
echo Usage: unrun_ollama.bat [-help]
echo Stops local Windows Ollama without uninstalling it or deleting models.
echo For Ubuntu, use ./unrun_ollama.sh on the Ubuntu server.
exit /b 0
