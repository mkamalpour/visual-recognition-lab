@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
 echo Activate your existing Conda base or ml environment first.
 pause
 exit /b 1
)
python -m streamlit run app.py --server.address 127.0.0.1
