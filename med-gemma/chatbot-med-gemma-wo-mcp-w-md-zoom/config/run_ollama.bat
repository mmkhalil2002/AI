@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem run_ollama.bat - Start, inspect, and test Ollama on Windows 11.
rem
rem HOW TO RUN (no options shows help; it does NOT start Ollama):
rem   run_ollama.bat                     Show help
rem   run_ollama.bat -remote             Start server for trusted LAN devices
rem   run_ollama.bat -local              Start server for this PC only
rem   run_ollama.bat -remote -test       Start and ask MedGemma "Hello"
rem   run_ollama.bat -test               Test an already running server
rem   run_ollama.bat -status             Check API and listening port
rem   run_ollama.bat -models             List installed models
rem   run_ollama.bat -pull qwen3:8b      Download another model
rem   run_ollama.bat -model qwen3:8b -test   Test another model
rem
rem OPTIONS:
rem   -h, -help, --help        Display this help
rem   -r, -remote, -cr        Bind to 0.0.0.0:11434 for LAN access
rem   -l, -local              Bind to 127.0.0.1:11434 for local access
rem   -t, -test               Generate a response to "Hello"
rem   -s, -status             Show API status and listening port
rem   -m, -models             List locally installed models
rem   -pull MODEL             Download MODEL (may be several GB)
rem   -model MODEL            Choose test model (default medgemma1.5)
rem
rem NOTES:
rem   * Requires Ollama installed on THIS Windows PC; for a Windows client
rem     accessing Ubuntu, use curl against Ubuntu's IP instead.
rem   * If Ollama is already running, this script will NOT change its bind.
rem     Stop the existing process/service before switching local/remote.
rem   * Ollama may auto-start with Windows; exit it from the tray first if
rem     changing its bind address.
rem   * -remote exposes an unauthenticated API. Restrict port 11434 to
rem     trusted LAN hosts with Windows Firewall. Do NOT expose to Internet.
rem   * This script never installs Ollama or deletes downloaded models.
rem ===========================================================================
if "%~1"=="" goto help
set "MODE="
set "DO_TEST=0"
set "DO_STATUS=0"
set "DO_MODELS=0"
set "PULL_MODEL="
set "MODEL=medgemma1.5"
if defined OLLAMA_MODEL set "MODEL=%OLLAMA_MODEL%"
set "PORT=11434"
if defined OLLAMA_PORT set "PORT=%OLLAMA_PORT%"
set "API=http://127.0.0.1:%PORT%"
set "OLLAMA_TEST_MODEL=%MODEL%"
set "OLLAMA_TEST_API=%API%"
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="-h" goto help
if /I "%~1"=="-help" goto help
if /I "%~1"=="--help" goto help
if /I "%~1"=="-r" goto remote
if /I "%~1"=="-remote" goto remote
if /I "%~1"=="-cr" goto remote
if /I "%~1"=="-connect_remotely" goto remote
if /I "%~1"=="--all-ips" goto remote
if /I "%~1"=="-l" goto local
if /I "%~1"=="-local" goto local
if /I "%~1"=="--local" goto local
if /I "%~1"=="-t" set "DO_TEST=1"& shift & goto parse
if /I "%~1"=="-test" set "DO_TEST=1"& shift & goto parse
if /I "%~1"=="--test" set "DO_TEST=1"& shift & goto parse
if /I "%~1"=="-s" set "DO_STATUS=1"& shift & goto parse
if /I "%~1"=="-status" set "DO_STATUS=1"& shift & goto parse
if /I "%~1"=="-m" set "DO_MODELS=1"& shift & goto parse
if /I "%~1"=="-models" set "DO_MODELS=1"& shift & goto parse
if /I "%~1"=="-pull" goto pullarg
if /I "%~1"=="-model" goto modelarg
echo [ERROR] Unknown option: %~1. Try -help. 1>&2
exit /b 1
:remote
if "%MODE%"=="local" goto conflict
set "MODE=remote"
shift
goto parse
:local
if "%MODE%"=="remote" goto conflict
set "MODE=local"
shift
goto parse
:conflict
echo [ERROR] Cannot combine -remote and -local. 1>&2
exit /b 1
:pullarg
shift
if "%~1"=="" goto missingarg
set "PULL_MODEL=%~1"
shift
goto parse
:modelarg
shift
if "%~1"=="" goto missingarg
set "MODEL=%~1"
set "OLLAMA_TEST_MODEL=%MODEL%"
shift
goto parse
:missingarg
echo [ERROR] Missing model name. 1>&2
exit /b 1
:parsed
if not defined MODE goto after_start
where ollama >nul 2>&1 || (echo [ERROR] Ollama is not installed or not on PATH. & exit /b 1)
call :ready
if not errorlevel 1 (
  echo [INFO] Ollama already running. Existing listening address unchanged.
  echo [INFO] Stop Ollama first if you need to change local/remote mode.
  goto after_start
)
if "%MODE%"=="remote" (set "OLLAMA_HOST=0.0.0.0:%PORT%") else (set "OLLAMA_HOST=127.0.0.1:%PORT%")
echo [INFO] Starting Ollama with OLLAMA_HOST=%OLLAMA_HOST%
rem Start detached, inheriting OLLAMA_HOST; keep logs in TEMP.
powershell -NoProfile -Command "try { Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $env:TEMP 'ollama-serve.log') -RedirectStandardError (Join-Path $env:TEMP 'ollama-serve-error.log') -ErrorAction Stop; exit 0 } catch { Write-Error $_; exit 1 }"
if errorlevel 1 exit /b 1
set /a ATTEMPTS=0
:wait_ready
call :ready
if not errorlevel 1 goto started
set /a ATTEMPTS+=1
if %ATTEMPTS% GEQ 30 (
 echo [ERROR] Ollama did not become ready. See %%TEMP%%\ollama-serve-error.log. 1>&2
 exit /b 1
)
timeout /t 1 /nobreak >nul
goto wait_ready
:started
echo [SUCCESS] Ollama API ready at %API%
if "%MODE%"=="remote" echo [SECURITY] Allow TCP %PORT% only from trusted LAN hosts.
:after_start
if "%DO_STATUS%"=="1" (
  call :ready
  if errorlevel 1 (echo [STATUS] API not responding at %API%) else (echo [STATUS] API responding at %API%)
  netstat -ano -p tcp | findstr /C:":%PORT% "
)
if "%DO_MODELS%"=="1" (
  call :require_ready || exit /b 1
  ollama list || exit /b 1
)
if defined PULL_MODEL (
  call :require_ready || exit /b 1
  echo [INFO] Pulling model %PULL_MODEL%
  ollama pull "%PULL_MODEL%" || exit /b 1
)
if "%DO_TEST%"=="1" (
  call :require_ready || exit /b 1
  echo [INFO] Asking %MODEL%: Hello
  rem PowerShell creates valid JSON and prints response + token statistics.
  powershell -NoProfile -Command "$ErrorActionPreference='Stop'; try { $body=@{model=$env:OLLAMA_TEST_MODEL;prompt='Hello';stream=$false}|ConvertTo-Json -Compress; $result=Invoke-RestMethod -Uri ($env:OLLAMA_TEST_API+'/api/generate') -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600; Write-Output '[RESPONSE]'; Write-Output $result.response; Write-Output ('[STATS] Input tokens: '+$result.prompt_eval_count+', output tokens: '+$result.eval_count) } catch { Write-Error $_; exit 1 }"
  if errorlevel 1 exit /b 1
)
exit /b 0
:ready
curl.exe -fsS --max-time 3 "%API%/api/tags" >nul 2>&1
exit /b %errorlevel%
:require_ready
call :ready
if errorlevel 1 (
 echo [ERROR] Ollama API not responding at %API%. Start with -local or -remote. 1>&2
 exit /b 1
)
exit /b 0
:help
echo run_ollama.bat - Ollama start, test and inspection utility for Windows
echo.
echo   run_ollama.bat                 Show help ^(default^)
echo   run_ollama.bat -remote         Start for remote LAN clients ^(alias -r, -cr^)
echo   run_ollama.bat -local          Start for local clients only ^(alias -l^)
echo   run_ollama.bat -remote -test   Start remotely and test MedGemma
echo   run_ollama.bat -test           Ask MedGemma "Hello" ^(alias -t^)
echo   run_ollama.bat -status         Check server ^(alias -s^)
echo   run_ollama.bat -models         List models ^(alias -m^)
echo   run_ollama.bat -pull MODEL     Download model
echo   run_ollama.bat -model MODEL -test  Test a different model
echo   run_ollama.bat -help           Display this help ^(alias -h^)
echo.
echo Remote access requires firewall rules restricting TCP 11434 to trusted LAN.
exit /b 0
