REM =============================================================================
REM Build a Standalone Executable with Nuitka
REM -----------------------------------------------------------------------------
REM Compile the application into a single standalone executable.
REM The command below:
REM   • Creates a local "nuitka_cache" folder in the current directory.
REM   • Automatically downloads required build tools.
REM   • Does not prompt for download confirmation.
REM   • Produces a standalone executable.
REM
REM Command:
REM
REM set NUITKA_CACHE_DIR=%CD%\nuitka_cache && ^
REM python -m nuitka --standalone --onefile ^
REM --assume-yes-for-downloads main.py
REM 
REM OR
REM python -m nuitka --standalone --onefile .\med-00-08-unknown30-cnn-128-256-512-1024-1744s-L5205-A9999-T8766.py
REM =============================================================================

@echo off
title Python Environment Installer

echo ============================================
echo Python Environment Installer
echo ============================================

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed.
    echo Download:
    echo https://python.org
    pause
    exit /b
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing basic packages...

python -m pip install ^
setuptools ^
wheel ^
virtualenv ^
numpy ^
scipy ^
pandas ^
matplotlib ^
opencv-python ^
pillow ^
requests ^
flask ^
pyyaml ^
psutil ^
tqdm ^
colorama ^
pyserial ^
cryptography ^
scikit-learn ^
joblib ^
onnx ^
onnxruntime ^
torch ^
torchvision ^
torchaudio ^
transformers ^
accelerate ^
sentencepiece ^
protobuf ^
huggingface_hub ^
nuitka ^
ordered-set ^
zstandard

echo.
echo ============================================
echo Installation Complete
echo ============================================

python -m nuitka --version

pause