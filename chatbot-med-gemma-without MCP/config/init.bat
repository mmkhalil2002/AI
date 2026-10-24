@echo off
setlocal EnableExtensions DisableDelayedExpansion
REM Windows setup for Medical AI project. Run this script from the config folder.
REM Installs Node.js, Python, Codex CLI; generates requirements.txt if absent.
REM Adds Node.js and npm global prefix permanently to the current USER PATH.
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
echo === Medical AI Windows initialization ===
echo Project: %PROJECT_DIR%
echo.

REM Find Node.js in typical locations if this CMD has a stale PATH.
if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
if exist "%LOCALAPPDATA%\Programs\nodejs\node.exe" set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"
where node.exe >nul 2>&1
if errorlevel 1 (
  echo [1/5] Installing Node.js LTS...
  where winget.exe >nul 2>&1
  if errorlevel 1 goto :manual_node
  winget install --id OpenJS.NodeJS.LTS --exact --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :manual_node
  if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
  where node.exe >nul 2>&1
  if errorlevel 1 (
    echo [ACTION] Open a new CMD and rerun this script after Node.js installation.
    exit /b 0
  )
)
where npm.cmd >nul 2>&1
if errorlevel 1 goto :manual_node
echo [1/5] Node.js and npm:
node --version
call npm.cmd --version
if errorlevel 1 goto :failed

REM IMPORTANT: Create npm's global prefix before npm list/install commands.
REM Use the CURRENT Windows account; never hardcode a username.
echo [2/5] Configuring npm global installation directory...
if not defined APPDATA (
  echo [ERROR] APPDATA environment variable is missing.
  goto :failed
)
set "NPM_PREFIX=%APPDATA%\npm"
if not exist "%NPM_PREFIX%" mkdir "%NPM_PREFIX%"
if errorlevel 1 goto :failed
call npm.cmd config set prefix "%NPM_PREFIX%"
if errorlevel 1 goto :failed
set "PATH=%NPM_PREFIX%;%PATH%"
echo npm global prefix: %NPM_PREFIX%

REM Permanently update USER PATH without using SETX (which can truncate PATH).
REM PowerShell reads existing user PATH and adds only missing entries.
echo [3/5] Saving Node.js and npm locations in user PATH...
for /f "delims=" %%I in ('where node.exe') do if not defined NODE_DIR for %%J in ("%%I") do set "NODE_DIR=%%~dpJ"
if not defined NODE_DIR (
  echo [ERROR] Unable to locate Node.js executable.
  goto :failed
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$paths = @($env:NODE_DIR, $env:NPM_PREFIX); $old = [Environment]::GetEnvironmentVariable('Path','User'); $entries = @($old -split ';' | Where-Object { $_ }); foreach ($p in $paths) { $p = $p.TrimEnd('\'); if ($entries -notcontains $p) { $entries += $p } }; [Environment]::SetEnvironmentVariable('Path', ($entries -join ';'), 'User')"
if errorlevel 1 (
  echo [WARNING] Could not save permanent user PATH. Current script PATH is still configured.
) else (
  echo User PATH updated. New CMD windows will use it.
)

REM Python installation. A new CMD may be needed after winget installs Python.
echo [4/5] Checking Python...
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
  if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
)
if not defined PYTHON_CMD (
  echo Installing Python 3.13...
  where winget.exe >nul 2>&1
  if errorlevel 1 goto :manual_python
  winget install --id Python.Python.3.13 --exact --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :manual_python
  echo [ACTION] Open a new CMD and rerun init.bat to finish setup.
  exit /b 0
)
%PYTHON_CMD% --version
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 %PYTHON_CMD% -m ensurepip --upgrade
%PYTHON_CMD% -m pip --version
if errorlevel 1 goto :failed

REM Install Codex after npm prefix has been created and configured.
echo [5/5] Installing/updating Codex CLI...
call npm.cmd install --global @openai/codex
if errorlevel 1 goto :failed
where codex.cmd >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Codex launcher not found. Expected: %NPM_PREFIX%\codex.cmd
  goto :failed
)
call codex.cmd --version
if errorlevel 1 goto :failed

REM Bootstrap pipreqs, then generate requirements.txt only if absent.
REM Scan the python source folder, not config or the entire project.
echo Checking Python project packages...
if not exist "%PROJECT_DIR%\requirements.txt" (
  if not exist "%PROJECT_DIR%\python" (
    echo [ERROR] Python source directory not found: "%PROJECT_DIR%\python"
    echo [ERROR] Cannot generate requirements.txt. Put your .py files in the python folder,
    echo [ERROR] or create requirements.txt manually in "%PROJECT_DIR%".
    goto :failed
  ) else (
    dir /b /s /a-d "%PROJECT_DIR%\python\*.py" 2>nul | findstr /r "." >nul
    if errorlevel 1 (
      echo [ERROR] No .py Python source files found under "%PROJECT_DIR%\python".
      echo [ERROR] Cannot generate requirements.txt. Add your Python program first,
      echo [ERROR] or create requirements.txt manually in "%PROJECT_DIR%".
      goto :failed
    )
    echo [INFO] Installing pipreqs to detect Python dependencies...
    %PYTHON_CMD% -m pip install pipreqs
    if errorlevel 1 goto :failed
    echo [INFO] Generating requirements.txt from python source files...
    %PYTHON_CMD% -m pipreqs.pipreqs "%PROJECT_DIR%\python" --savepath "%PROJECT_DIR%\requirements.txt" --force --mode no-pin --ignore .venv,venv
    if errorlevel 1 (
      echo [ERROR] Automatic dependency detection failed. Check the Python source files.
      echo [ERROR] Alternatively create requirements.txt manually in "%PROJECT_DIR%".
      goto :failed
    )
    if not exist "%PROJECT_DIR%\requirements.txt" (
      echo [ERROR] pipreqs completed but did not create requirements.txt.
      goto :failed
    )
  )
)

if exist "%PROJECT_DIR%\requirements.txt" (
  %PYTHON_CMD% -m pip install -r "%PROJECT_DIR%\requirements.txt"
  if errorlevel 1 goto :failed
) else (
  echo [WARNING] requirements.txt not found; Python packages skipped.
)
echo.
REM Optional Ollama installation on THIS Windows machine only.
echo.
choice /C YN /N /M "Configure Ollama and AI models on THIS Windows computer? [Y/N] "
if errorlevel 2 goto :models_done
choice /C YN /N /M "Install Ollama locally if missing? [Y/N] "
if errorlevel 2 goto :local_models
where ollama.exe >nul 2>&1
if errorlevel 1 (
  where winget.exe >nul 2>&1
  if errorlevel 1 (
    echo [WARNING] winget unavailable. Install Ollama from https://ollama.com/download/windows
  ) else (
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 echo [WARNING] Ollama installation failed.
    set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
  )
)
:local_models
where ollama.exe >nul 2>&1
if errorlevel 1 (
  echo [WARNING] Ollama not found; model downloads skipped.
  goto :models_done
)
ollama list >nul 2>&1
if errorlevel 1 (
  echo [WARNING] Ollama is not running. Start the Ollama application and rerun init.
  goto :models_done
)
choice /C YN /N /M "Download MedGemma 1.5 on THIS computer? [Y/N] "
if errorlevel 2 goto :local_qwen
ollama pull medgemma1.5
if errorlevel 1 echo [WARNING] MedGemma download failed.
:local_qwen
choice /C YN /N /M "Download Qwen3:8B on THIS computer? [Y/N] "
if errorlevel 2 goto :models_done
ollama pull qwen3:8b
if errorlevel 1 echo [WARNING] Qwen3 download failed.
:models_done
echo [SUCCESS] Setup completed.
echo Codex CLI installed at: %NPM_PREFIX%\codex.cmd
echo Open a NEW CMD window and run:
echo   cd /d "%PROJECT_DIR%"
echo   codex --no-daemon
echo.
exit /b 0

:manual_node
echo [ERROR] Install Node.js LTS from https://nodejs.org/en/download
exit /b 1
:manual_python
echo [ERROR] Install Python from https://www.python.org/downloads/windows/
exit /b 1
:failed
echo [ERROR] Initialization failed. See details above.
exit /b 1
