@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Nuitka Build - ENV Configuration - Final EXE Only

REM ============================================================================
REM NUITKA WINDOWS ONEFILE BUILD
REM ============================================================================
REM
REM Configuration is loaded from:
REM     .env
REM
REM The .env file must be in the directory where this BAT file is executed.
REM
REM Required variables:
REM     BASE_DIRECTORY
REM     INPUT_FILE
REM     OUTPUT_FILE
REM
REM This script:
REM   - Uses no interactive input.
REM   - Starts from the current directory.
REM   - Uses a short temporary source name: app.py.
REM   - Uses a short temporary output name: app.exe.
REM   - Deletes old _no, _nw, *.build, *.dist and *.onefile-build folders.
REM   - Deletes its temporary build tree after success or failure.
REM   - Leaves only the final EXE in Nuitka_Output after success.
REM ============================================================================

set "LAUNCH_DIRECTORY=%CD%"
set "ENV_FILE=%LAUNCH_DIRECTORY%\.env"
set "BUILD_RESULT=1"
set "BUILD_SUCCEEDED=0"
set "SHORT_BUILD_ROOT="

cls
echo.
echo ================================================================
echo      NUITKA ENV BUILD - FINAL EXECUTABLE ONLY
echo ================================================================
echo.

REM ============================================================================
REM LOAD .ENV
REM ============================================================================

if not exist "%ENV_FILE%" (
    echo ERROR: The .env file was not found:
    echo "%ENV_FILE%"
    goto :FINAL_FAILURE
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    set "ENV_KEY=%%A"
    set "ENV_VALUE=%%B"

    for /f "tokens=* delims= " %%K in ("!ENV_KEY!") do set "ENV_KEY=%%K"
    for /f "tokens=* delims= " %%V in ("!ENV_VALUE!") do set "ENV_VALUE=%%V"

    if defined ENV_KEY set "!ENV_KEY!=!ENV_VALUE!"
)

if not defined BASE_DIRECTORY (
    echo ERROR: BASE_DIRECTORY is missing from .env
    goto :FINAL_FAILURE
)

if not defined INPUT_FILE (
    echo ERROR: INPUT_FILE is missing from .env
    goto :FINAL_FAILURE
)

if not defined OUTPUT_FILE (
    echo ERROR: OUTPUT_FILE is missing from .env
    goto :FINAL_FAILURE
)

REM ============================================================================
REM RESOLVE BASE DIRECTORY
REM ============================================================================

pushd "%LAUNCH_DIRECTORY%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot access the launch directory.
    goto :FINAL_FAILURE
)

pushd "%BASE_DIRECTORY%" >nul 2>&1
if errorlevel 1 (
    popd >nul 2>&1
    echo ERROR: Cannot access BASE_DIRECTORY:
    echo "%BASE_DIRECTORY%"
    goto :FINAL_FAILURE
)

set "ROOT_DIR=%CD%"
popd >nul 2>&1
popd >nul 2>&1

REM ============================================================================
REM RESOLVE INPUT FILE
REM ============================================================================

set "INPUT_FILE_NORMALIZED=%INPUT_FILE%"
if "%INPUT_FILE_NORMALIZED:~0,1%"=="\" set "INPUT_FILE_NORMALIZED=%INPUT_FILE_NORMALIZED:~1%"
if "%INPUT_FILE_NORMALIZED:~0,1%"=="/" set "INPUT_FILE_NORMALIZED=%INPUT_FILE_NORMALIZED:~1%"

set "SCRIPT_FULL_PATH=%ROOT_DIR%\%INPUT_FILE_NORMALIZED%"

for %%I in ("%SCRIPT_FULL_PATH%") do (
    set "SCRIPT_FULL_PATH=%%~fI"
    set "SCRIPT_NAME=%%~nI"
    set "SCRIPT_EXT=%%~xI"
)

if not exist "%SCRIPT_FULL_PATH%" (
    echo ERROR: INPUT_FILE was not found:
    echo "%SCRIPT_FULL_PATH%"
    goto :FINAL_FAILURE
)

if /I not "%SCRIPT_EXT%"==".py" (
    echo ERROR: INPUT_FILE must be a Python .py file.
    goto :FINAL_FAILURE
)

REM ============================================================================
REM NORMALIZE OUTPUT NAME
REM ============================================================================

set "OUTPUT_NAME=%OUTPUT_FILE%"
if "%OUTPUT_NAME:~0,1%"=="\" set "OUTPUT_NAME=%OUTPUT_NAME:~1%"
if "%OUTPUT_NAME:~0,1%"=="/" set "OUTPUT_NAME=%OUTPUT_NAME:~1%"

for %%I in ("%OUTPUT_NAME%") do set "OUTPUT_NAME=%%~nxI"

if /I "%OUTPUT_NAME:~-3%"==".py" (
    set "OUTPUT_NAME=%OUTPUT_NAME:~0,-3%.exe"
)

if /I not "%OUTPUT_NAME:~-4%"==".exe" (
    set "OUTPUT_NAME=%OUTPUT_NAME%.exe"
)

if "%OUTPUT_NAME%"==".exe" (
    set "OUTPUT_NAME=%SCRIPT_NAME%.exe"
)

REM ============================================================================
REM FIND PYTHON
REM ============================================================================

set "PYTHON_EXE="

where py >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq delims=" %%P in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do (
        set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do (
            set "PYTHON_EXE=%%P"
        )
    )
)

if not defined PYTHON_EXE (
    echo ERROR: Python 3 was not found.
    goto :FINAL_FAILURE
)

REM ============================================================================
REM PATHS
REM ============================================================================

set "FINAL_OUTPUT=%ROOT_DIR%\Nuitka_Output"
set "FINAL_EXE=%FINAL_OUTPUT%\%OUTPUT_NAME%"

set "SHORT_BUILD_ROOT=%TEMP%\NB_%RANDOM%_%RANDOM%"
set "WORK_DIR=%SHORT_BUILD_ROOT%\w"
set "TEMP_DIR=%SHORT_BUILD_ROOT%\t"
set "CACHE_DIR=%SHORT_BUILD_ROOT%\c"
set "BUILD_OUTPUT=%SHORT_BUILD_ROOT%\o"
set "STAGE_DIR=%SHORT_BUILD_ROOT%\s"
set "STAGED_SCRIPT=%STAGE_DIR%\app.py"
set "INTERNAL_EXE=%BUILD_OUTPUT%\app.exe"
set "RUNNER=%WORK_DIR%\run.ps1"
set "LOG_FILE=%SHORT_BUILD_ROOT%\build.log"
set "REPORT_FILE=%SHORT_BUILD_ROOT%\report.xml"

echo Configuration:
echo ---------------------------------------------------------------
echo Base directory : "%ROOT_DIR%"
echo Input file     : "%SCRIPT_FULL_PATH%"
echo Output file    : "%FINAL_EXE%"
echo Python         : "%PYTHON_EXE%"
echo ---------------------------------------------------------------
echo.

REM ============================================================================
REM CLEAN OLD BUILD ARTIFACTS
REM ============================================================================

call :CLEAN_LEGACY_ARTIFACTS

if exist "%SHORT_BUILD_ROOT%" rmdir /S /Q "%SHORT_BUILD_ROOT%" >nul 2>&1
if exist "%FINAL_OUTPUT%" rmdir /S /Q "%FINAL_OUTPUT%" >nul 2>&1

mkdir "%WORK_DIR%" >nul 2>&1
mkdir "%TEMP_DIR%" >nul 2>&1
mkdir "%CACHE_DIR%" >nul 2>&1
mkdir "%BUILD_OUTPUT%" >nul 2>&1
mkdir "%STAGE_DIR%" >nul 2>&1
mkdir "%FINAL_OUTPUT%" >nul 2>&1

if not exist "%WORK_DIR%" goto :DIRECTORY_FAILURE
if not exist "%TEMP_DIR%" goto :DIRECTORY_FAILURE
if not exist "%CACHE_DIR%" goto :DIRECTORY_FAILURE
if not exist "%BUILD_OUTPUT%" goto :DIRECTORY_FAILURE
if not exist "%STAGE_DIR%" goto :DIRECTORY_FAILURE
if not exist "%FINAL_OUTPUT%" goto :DIRECTORY_FAILURE

copy /Y "%SCRIPT_FULL_PATH%" "%STAGED_SCRIPT%" >nul
if errorlevel 1 (
    echo ERROR: Could not create the short temporary source file.
    goto :FINAL_FAILURE
)

REM Preserve imports and files relative to the original project directory.
set "PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%"
set "TEMP=%TEMP_DIR%"
set "TMP=%TEMP_DIR%"
set "TMPDIR=%TEMP_DIR%"
set "PYTHONDONTWRITEBYTECODE=1"
set "NUITKA_CACHE_DIR=%CACHE_DIR%"
set "NUITKA_CACHE_DIR_DOWNLOADS=%CACHE_DIR%\downloads"
set "NUITKA_CACHE_DIR_CCACHE=%CACHE_DIR%\ccache"

REM ============================================================================
REM INSTALL / UPDATE REQUIREMENTS
REM ============================================================================

echo Installing or updating Nuitka build requirements...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel nuitka ordered-set zstandard
if errorlevel 1 (
    echo ERROR: Nuitka requirements could not be installed.
    goto :FINAL_FAILURE
)

if exist "%ROOT_DIR%\requirements.txt" (
    echo Installing requirements.txt...
    "%PYTHON_EXE%" -m pip install -r "%ROOT_DIR%\requirements.txt"
    if errorlevel 1 (
        echo ERROR: requirements.txt installation failed.
        goto :FINAL_FAILURE
    )
)

REM ============================================================================
REM CHECK SOURCE SYNTAX
REM ============================================================================

echo.
echo Checking Python source syntax...
set "SOURCE_TO_CHECK=%SCRIPT_FULL_PATH%"

"%PYTHON_EXE%" -X utf8 -c "import os,tokenize; p=os.environ['SOURCE_TO_CHECK']; f=tokenize.open(p); s=f.read(); f.close(); compile(s,p,'exec'); print('Python syntax check completed successfully.')"
if errorlevel 1 (
    echo ERROR: The Python source contains a syntax error.
    goto :FINAL_FAILURE
)

REM ============================================================================
REM CREATE POWERSHELL RUNNER
REM ============================================================================

> "%RUNNER%" echo $ErrorActionPreference = 'Continue'
>>"%RUNNER%" echo $a = [System.Collections.Generic.List[string]]::new()
>>"%RUNNER%" echo $a.Add('-m')
>>"%RUNNER%" echo $a.Add('nuitka')
>>"%RUNNER%" echo $a.Add('--onefile')
>>"%RUNNER%" echo $a.Add('--follow-imports')
>>"%RUNNER%" echo $a.Add('--msvc=latest')
>>"%RUNNER%" echo $a.Add('--assume-yes-for-downloads')
>>"%RUNNER%" echo $a.Add('--show-progress')
>>"%RUNNER%" echo $a.Add('--show-memory')
>>"%RUNNER%" echo $a.Add('--jobs=' + [Environment]::ProcessorCount)
>>"%RUNNER%" echo $a.Add('--output-dir=' + $env:BUILD_OUTPUT)
>>"%RUNNER%" echo $a.Add('--output-filename=app.exe')
>>"%RUNNER%" echo $a.Add('--report=' + $env:REPORT_FILE)
>>"%RUNNER%" echo $folders = @('models','model','data','assets','images','weights','config','configs')
>>"%RUNNER%" echo foreach ($name in $folders) {
>>"%RUNNER%" echo     $src = Join-Path $env:ROOT_DIR $name
>>"%RUNNER%" echo     if (Test-Path -LiteralPath $src -PathType Container) {
>>"%RUNNER%" echo         Write-Host ('Including data directory: ' + $src)
>>"%RUNNER%" echo         $a.Add('--include-data-dir=' + $src + '=' + $name)
>>"%RUNNER%" echo     }
>>"%RUNNER%" echo }
>>"%RUNNER%" echo $a.Add($env:STAGED_SCRIPT)
>>"%RUNNER%" echo Write-Host ''
>>"%RUNNER%" echo Write-Host ('Temporary source: ' + $env:STAGED_SCRIPT)
>>"%RUNNER%" echo Write-Host ('Internal output: ' + $env:INTERNAL_EXE)
>>"%RUNNER%" echo Write-Host ''
>>"%RUNNER%" echo ^& $env:PYTHON_EXE $a 2^>^&1 ^| Tee-Object -FilePath $env:LOG_FILE
>>"%RUNNER%" echo $code = $LASTEXITCODE
>>"%RUNNER%" echo exit $code

if not exist "%RUNNER%" (
    echo ERROR: Could not create the PowerShell build runner.
    goto :FINAL_FAILURE
)

REM ============================================================================
REM BUILD
REM ============================================================================

echo.
echo ================================================================
echo STARTING NUITKA BUILD
echo ================================================================
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%RUNNER%"
set "BUILD_RESULT=%ERRORLEVEL%"

if not "%BUILD_RESULT%"=="0" goto :BUILD_FAILURE
if not exist "%INTERNAL_EXE%" (
    echo ERROR: Nuitka returned success but app.exe was not found.
    goto :BUILD_FAILURE
)

REM ============================================================================
REM COPY FINAL EXE
REM ============================================================================

copy /Y "%INTERNAL_EXE%" "%FINAL_EXE%" >nul
if errorlevel 1 (
    echo ERROR: The final executable could not be copied.
    goto :BUILD_FAILURE
)

set "BUILD_SUCCEEDED=1"

REM ============================================================================
REM SUCCESS CLEANUP
REM ============================================================================

call :REMOVE_ALL_TEMPORARY_FILES

if not exist "%FINAL_EXE%" (
    echo ERROR: The executable is missing after cleanup.
    goto :FINAL_FAILURE
)

echo.
echo ================================================================
echo BUILD SUCCESSFUL
echo ================================================================
echo.
echo Only this executable remains:
echo "%FINAL_EXE%"
echo.

pause
endlocal
exit /b 0

REM ============================================================================
REM BUILD FAILURE
REM ============================================================================

:BUILD_FAILURE
echo.
echo ================================================================
echo BUILD FAILED
echo ================================================================
echo.
echo Nuitka exit code: %BUILD_RESULT%
echo.

if exist "%LOG_FILE%" (
    echo Important error lines:
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Select-String -LiteralPath $env:LOG_FILE -Pattern 'CopyFile2|WinError 3|FileNotFoundError|FATAL:|ERROR:' -Context 10,10 | ForEach-Object { $_.ToString() }"
)

call :REMOVE_BUILD_TEMP_ONLY

echo.
echo All temporary build directories were removed.
echo No executable was produced.
echo.
pause
endlocal
exit /b 1

REM ============================================================================
REM GENERAL FAILURE
REM ============================================================================

:DIRECTORY_FAILURE
echo ERROR: One or more temporary directories could not be created.

:FINAL_FAILURE
call :REMOVE_BUILD_TEMP_ONLY
echo.
echo Build stopped.
echo Temporary build directories were removed.
echo.
pause
endlocal
exit /b 1

REM ============================================================================
REM CLEANUP SUBROUTINES
REM ============================================================================

:CLEAN_LEGACY_ARTIFACTS
REM Remove old folders created by earlier versions of this script.
if exist "%ROOT_DIR%\_no" rmdir /S /Q "%ROOT_DIR%\_no" >nul 2>&1
if exist "%ROOT_DIR%\_nw" rmdir /S /Q "%ROOT_DIR%\_nw" >nul 2>&1
if exist "%ROOT_DIR%\Nuitka_Work" rmdir /S /Q "%ROOT_DIR%\Nuitka_Work" >nul 2>&1

for /D %%D in ("%ROOT_DIR%\*.build") do (
    if exist "%%~fD" rmdir /S /Q "%%~fD" >nul 2>&1
)

for /D %%D in ("%ROOT_DIR%\*.dist") do (
    if exist "%%~fD" rmdir /S /Q "%%~fD" >nul 2>&1
)

for /D %%D in ("%ROOT_DIR%\*.onefile-build") do (
    if exist "%%~fD" rmdir /S /Q "%%~fD" >nul 2>&1
)

exit /b 0

:REMOVE_BUILD_TEMP_ONLY
if defined SHORT_BUILD_ROOT (
    if exist "%SHORT_BUILD_ROOT%" rmdir /S /Q "%SHORT_BUILD_ROOT%" >nul 2>&1
)

call :CLEAN_LEGACY_ARTIFACTS
exit /b 0

:REMOVE_ALL_TEMPORARY_FILES
call :REMOVE_BUILD_TEMP_ONLY

REM Delete every subdirectory in Nuitka_Output.
for /D %%D in ("%FINAL_OUTPUT%\*") do (
    if exist "%%~fD" rmdir /S /Q "%%~fD" >nul 2>&1
)

REM Delete every file except the final EXE.
for %%F in ("%FINAL_OUTPUT%\*") do (
    if /I not "%%~fF"=="%FINAL_EXE%" (
        del /F /Q "%%~fF" >nul 2>&1
    )
)

exit /b 0
