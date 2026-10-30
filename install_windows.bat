@echo off
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Installation complete. Run start_windows.bat
pause
exit /b 0
:error
echo Installation failed. Install Python 3.10 or newer from python.org with the Python launcher.
pause
exit /b 1
