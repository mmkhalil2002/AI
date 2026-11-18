@echo off
setlocal
set "APP=%~dp0..\python\night_guard_server.py"
py.exe -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 goto launcher

set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PYTHON%" goto interpreter
set "PYTHON=%ProgramFiles%\Python312\python.exe"
if exist "%PYTHON%" goto interpreter
set "PYTHON=python.exe"
"%PYTHON%" -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 goto interpreter
echo Python is unavailable. Run config\init-server-windows.ps1 first. 1>&2
exit /b 1

:launcher
py.exe -3.12 "%APP%"
exit /b %errorlevel%

:interpreter
"%PYTHON%" "%APP%"
exit /b %errorlevel%
