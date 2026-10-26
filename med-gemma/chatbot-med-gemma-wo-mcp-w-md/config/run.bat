@echo off
setlocal EnableExtensions DisableDelayedExpansion
REM Launch the named MedGemma patient GUI.
REM Ollama address is configured by the Python application in python/.env.
set "PYTHON_DIR=%~dp0..\python"
if not exist "%PYTHON_DIR%\" (
  echo [ERROR] Python directory not found: "%PYTHON_DIR%"
  exit /b 1
)
set "MAIN_PY=%PYTHON_DIR%\client_medgemma_patients.py"
if not exist "%MAIN_PY%" (
  echo [ERROR] MedGemma application not found: "%MAIN_PY%"
  exit /b 1
)
set "PYTHON_EXE="
set "PYTHON_ARGS="
where python >nul 2>&1
if not errorlevel 1 python -c "import sys, tkinter; print(sys.executable); sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_EXE=python"
  goto :python_ready
)
where py >nul 2>&1
if not errorlevel 1 py -3 -c "import sys, tkinter; print(sys.executable); sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_EXE=py"
  set "PYTHON_ARGS=-3"
  goto :python_ready
)
REM Windows Store aliases can hide a real per-user install. Probe the standard
REM per-user locations without changing PATH or installing anything.
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
  if exist "%%~fD\python.exe" (
    "%%~fD\python.exe" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
    if not errorlevel 1 (
      set "PYTHON_EXE=%%~fD\python.exe"
      goto :python_ready
    )
  )
)
REM Microsoft Store Python and some managed installs are registered under
REM PythonCore but are not exposed through PATH or the ``py`` launcher.
REM Check the registration before reporting that Python is unavailable.
for %%V in (3.13 3.12 3.11 3.10 3.9) do (
  for %%K in ("HKCU\Software\Python\PythonCore\%%V\InstallPath" "HKLM\Software\Python\PythonCore\%%V\InstallPath") do (
    for /f "tokens=2,*" %%A in ('reg query %%K /ve 2^>nul ^| find /i "REG_SZ"') do (
      if exist "%%B\python.exe" (
        "%%B\python.exe" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
        if not errorlevel 1 (
          set "PYTHON_EXE=%%B\python.exe"
          goto :python_ready
        )
      )
      if exist "%%B\python%%V.exe" (
        "%%B\python%%V.exe" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
        if not errorlevel 1 (
          set "PYTHON_EXE=%%B\python%%V.exe"
          goto :python_ready
        )
      )
    )
  )
)
echo [ERROR] A working Python 3.9+ interpreter with Tkinter is unavailable.
echo [ACTION] Install Python 3.9 or newer with Tcl/Tk support, then retry. If the Windows Store alias intercepted Python, disable it in App execution aliases.
exit /b 1

:python_ready
pushd "%PYTHON_DIR%" || exit /b 1
echo [INFO] Running: "%MAIN_PY%"
if defined PYTHON_ARGS (
  echo [INFO] Python command: "%PYTHON_EXE%" %PYTHON_ARGS%
  "%PYTHON_EXE%" %PYTHON_ARGS% "%MAIN_PY%"
) else (
  echo [INFO] Python command: "%PYTHON_EXE%"
  "%PYTHON_EXE%" "%MAIN_PY%"
)
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" echo [ERROR] Python program exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
