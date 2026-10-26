@echo off
setlocal EnableExtensions
REM Install Python (if needed), create venv, then call config.py.
cd /d "%~dp0"
set "PYTHON_CMD="
py -3.12 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.12"
if not defined PYTHON_CMD (
    py -3 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    python -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    where winget >nul 2>&1
    if errorlevel 1 (
        echo winget is not available. Install Python 3.12 from python.org and rerun.
        exit /b 1
    )
    echo Installing Python 3.12 with winget...
    winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 exit /b 1
    REM Use the installed interpreter directly; PATH may not refresh until next shell.
    set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
    if not exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        echo Python installed but interpreter was not found. Open a new terminal and rerun.
        exit /b 1
    )
)
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" config.py
if errorlevel 1 exit /b 1
echo Installation complete.
exit /b 0
