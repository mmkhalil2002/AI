@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if not errorlevel 1 (
    py -3 generate_patient_data.py
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python 3 was not found. Install Python 3 and try again.
        pause
        exit /b 1
    )
    python generate_patient_data.py
)

set "RUN_EXIT=%ERRORLEVEL%"
echo.
if not "%RUN_EXIT%"=="0" echo [ERROR] The importer stopped with exit code %RUN_EXIT%.
pause
exit /b %RUN_EXIT%
