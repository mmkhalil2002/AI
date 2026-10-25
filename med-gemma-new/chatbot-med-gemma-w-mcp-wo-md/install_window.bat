
@echo off
setlocal EnableExtensions

REM ============================================================
REM FILE: install_windows.bat
REM
REM PURPOSE:
REM   Install Python if necessary, create a virtual environment,
REM   and execute config.py to install Medical MCP dependencies.
REM
REM REQUIREMENTS:
REM   Windows with winget available if Python is not installed.
REM   Internet access for downloading Python and packages.
REM
REM PROCESSING:
REM   1. Locate Python 3.11 or later.
REM   2. Install Python through winget if missing.
REM   3. Create a project virtual environment.
REM   4. Upgrade pip.
REM   5. Run config.py.
REM   6. Report installation status.
REM ============================================================

REM STEP 1: Change to the directory containing this script.
cd /d "%~dp0"

REM STEP 2: Check whether the Python launcher can find
REM a compatible Python interpreter.
set "PYTHON_CMD="

py -3.11 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"

REM STEP 3: If the launcher did not work, try python.exe.
if not defined PYTHON_CMD (
    python -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

REM STEP 4: Install Python if no compatible interpreter exists.
if not defined PYTHON_CMD (
    echo Python 3.11 or later was not found.
    echo Installing Python using winget...

    REM Confirm that Windows Package Manager is available.
    where winget >nul 2>&1
    if errorlevel 1 (
        echo ERROR: winget is unavailable.
        echo Install Python from https://www.python.org/downloads/
        exit /b 1
    )

    REM Install Python 3.12 for the current user.
    winget install --id Python.Python.3.12 --exact ^
        --source winget ^
        --scope user ^
        --accept-package-agreements ^
        --accept-source-agreements

    if errorlevel 1 (
        echo ERROR: Python installation failed.
        exit /b 1
    )

    REM Try the Python launcher after installation.
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3.12"

    REM A new terminal may be needed for PATH updates.
    if not defined PYTHON_CMD (
        echo Python may be installed, but is not yet visible.
        echo Close this terminal, open a new one, and rerun.
        exit /b 1
    )
)

REM STEP 5: Create the project virtual environment.
REM This isolates MCP packages from other Python projects.
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :failed
)

REM STEP 6: Upgrade pip in the virtual environment.
echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed

REM STEP 7: Execute the shared Python configuration routine.
echo Installing Medical MCP packages...
".venv\Scripts\python.exe" config.py
if errorlevel 1 goto :failed

REM STEP 8: Display successful installation.
echo.
echo Medical MCP installation completed successfully.
echo To run the X-ray server:
echo .venv\Scripts\python.exe xray_server.py
exit /b 0

:failed
echo.
echo ERROR: Medical MCP installation failed.
exit /b 1
