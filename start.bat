@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the Python environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Install Python 3.12, then run this file again.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -c "import flask, cv2, numpy, PIL; from importlib.util import find_spec; assert all(find_spec(name) for name in ['matplotlib', 'scipy', 'pywt', 'skimage']), 'Chapter dependencies are missing'"
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m ensurepip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
echo.
echo PixelLab is starting at http://127.0.0.1:5000
echo Keep this window open. Press Ctrl+C to stop the app.
echo.
".venv\Scripts\python.exe" "python_code\code_01_web_app.py" --open-browser
pause
