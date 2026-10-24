@echo off
setlocal EnableExtensions
rem MedGemma project cleanup for Windows.
rem Removes global Codex CLI and this project's .venv only.
rem Keeps Python, Node.js, npm, PATH, pipreqs, requirements.txt, and project data.
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
echo === MedGemma Windows uninitialization ===
echo Project: %PROJECT_ROOT%
echo.
echo This removes the globally installed OpenAI Codex CLI and the
 echo project-local .venv, if present.
echo It does NOT remove Python, Node.js, npm, pipreqs, PATH entries,
echo requirements.txt, your code, patient records, or Ollama models.
echo.
choice /C YN /N /M "Continue? [Y/N]: "
if errorlevel 2 goto cancelled
set "PATH=%ProgramFiles%\nodejs;%APPDATA%\npm;%PATH%"
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo [WARNING] npm not found. Skipping Codex uninstall.
) else (
  echo [1/2] Removing Codex CLI from current npm global prefix...
  call npm uninstall -g @openai/codex
  if errorlevel 1 echo [WARNING] Codex removal failed. Check npm configuration.
)
if exist "%PROJECT_ROOT%\.venv\" (
  echo [2/2] Removing project virtual environment...
  rmdir /S /Q "%PROJECT_ROOT%\.venv"
  if exist "%PROJECT_ROOT%\.venv\" echo [WARNING] Could not fully remove .venv. Close Python processes and retry.
) else (
  echo [2/2] No project .venv found.
)
echo.
echo Finished. requirements.txt and all project files were preserved.
goto end
:cancelled
echo Cancelled. No changes made.
:end
endlocal
