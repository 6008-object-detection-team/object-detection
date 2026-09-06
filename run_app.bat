@echo off
setlocal
cd /d "%~dp0"
REM Prefer an explicit interpreter, then the environment selected by the user.
if defined OBJECT_DETECTION_PYTHON (
    set "APP_PYTHON=%OBJECT_DETECTION_PYTHON%"
) else if defined VIRTUAL_ENV (
    set "APP_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
) else if defined CONDA_PREFIX (
    set "APP_PYTHON=%CONDA_PREFIX%\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "D:\anaconda3\envs\pytorch\python.exe" (
    set "APP_PYTHON=D:\anaconda3\envs\pytorch\python.exe"
) else (
    set "APP_PYTHON=python"
)
if /i "%~1"=="--check" (
    "%APP_PYTHON%" "%~dp0check_environment.py"
    exit /b
)
"%APP_PYTHON%" "%~dp0main.py"
pause
endlocal
