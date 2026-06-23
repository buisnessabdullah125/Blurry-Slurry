@echo off
setlocal

"%~dp0..\.venv\Scripts\python.exe" -m pip install pyinstaller
"%~dp0..\.venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean build_exe.spec

endlocal