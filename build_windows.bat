@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto error
.venv\Scripts\python.exe -m pip install pyinstaller==6.16.0
if errorlevel 1 goto error
.venv\Scripts\python.exe audio_cleaner.py --download-model --model-dir models
if errorlevel 1 goto error
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --windowed --name AudioCleaner --collect-all imageio_ffmpeg --collect-all sherpa_onnx --copy-metadata sherpa-onnx-core --add-data "models;models" --add-data "THIRD_PARTY_NOTICES.md;." --add-data "licenses;licenses" audio_cleaner.py
if errorlevel 1 goto error
echo Ready: dist\AudioCleaner.exe
pause
exit /b 0
:error
echo Build failed. Run install_windows.bat first. Build on 64-bit Windows.
pause
exit /b 1
