@echo off
setlocal EnableExtensions DisableDelayedExpansion
REM Launch the first .py file (alphabetical order) in ../python.
REM Ollama address is configured by the Python application in python/.env.
set "PYTHON_DIR=%~dp0..\python"
if not exist "%PYTHON_DIR%\" (
  echo [ERROR] Python directory not found: "%PYTHON_DIR%"
  exit /b 1
)
set "MAIN_PY="
for /f "delims=" %%F in ('dir /b /a-d /on "%PYTHON_DIR%\*.py" 2^>nul') do if not defined MAIN_PY set "MAIN_PY=%%F"
if not defined MAIN_PY (
  echo [ERROR] No .py files found in "%PYTHON_DIR%".
  exit /b 1
)
where python >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_CMD=python"
) else (
  where py >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    exit /b 1
  )
  set "PYTHON_CMD=py -3"
)
cd /d "%PYTHON_DIR%" || exit /b 1
echo [INFO] Running: "%MAIN_PY%"
echo [INFO] Working directory: "%CD%"
%PYTHON_CMD% "%MAIN_PY%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [ERROR] Python program exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
