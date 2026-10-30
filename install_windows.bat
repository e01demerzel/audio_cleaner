@echo off
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
.venv\Scripts\python.exe audio_cleaner.py --download-model
if errorlevel 1 goto error
echo Installation complete. Run start_windows.bat
pause
exit /b 0
:error
echo Installation failed. Use 64-bit Python 3.10 or newer with the Python launcher.
echo Model download requires access to GitHub. Read the error above and retry.
pause
exit /b 1
