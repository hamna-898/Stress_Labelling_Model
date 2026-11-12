@echo off
REM Step 5: create the video environment (py-feat) and check it.
REM Separate from audio_env because py-feat needs a newer PyTorch (2.11+).
cd /d "%~dp0"

echo === Checking Python 3.11 ===
py -3.11 --version
if errorlevel 1 ( echo Python 3.11 not found. & pause & exit /b 1 )

if not exist video_env (
  echo === Creating video_env ===
  py -3.11 -m venv video_env
)
call video_env\Scripts\activate.bat

echo === Installing PyTorch 2.11+ with CUDA 12.6 (large download, ~2.5-3 GB) ===
python -m pip install --upgrade pip
pip install "torch>=2.11" torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 ( echo PyTorch install failed - run this file again. & pause & exit /b 1 )

echo === Installing py-feat and helpers ===
pip install -r requirements-video.txt
if errorlevel 1 ( echo Package install failed - run this file again. & pause & exit /b 1 )

echo === Video environment check ===
python src\check_video_env.py
pause
