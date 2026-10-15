@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo         STARTING E.D.I.T.H. NEURAL NETWORK TRAINING      
echo =========================================================

set "PYTHON_EXE=C:\Users\Aditya Sawant\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0..\train.py"

echo.
pause
