@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe audio_cleaner.py
if errorlevel 1 pause
