@echo off
REM Launch with the verified CUDA/YOLOE Conda environment, not the system Python.
cd /d "%~dp0"
"D:\anaconda3\envs\pytorch\python.exe" "%~dp0main.py"
pause
