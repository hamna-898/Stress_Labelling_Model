@echo off
REM Step 1: create the audio environment and check the GPU.
cd /d "%~dp0"

echo === Checking Python 3.11 ===
py -3.11 --version
if errorlevel 1 (
  echo Python 3.11 not found. Install it from python.org and run this again.
  pause
  exit /b 1
)

if not exist audio_env (
  echo === Creating audio_env ===
  py -3.11 -m venv audio_env
)
call audio_env\Scripts\activate.bat

echo === Installing PyTorch with CUDA (large download, ~2.5 GB) ===
python -m pip install --upgrade pip
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 ( echo PyTorch install failed. & pause & exit /b 1 )

echo === Installing the other audio packages ===
pip install -r requirements-audio.txt
if errorlevel 1 ( echo Package install failed. & pause & exit /b 1 )

echo === GPU check ===
python src\check_gpu.py
pause
