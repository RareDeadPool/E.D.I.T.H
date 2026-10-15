@echo off
setlocal enabledelayedexpansion

set "PYTHON_EXE="
if exist "C:\Users\Aditya Sawant\AppData\Local\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=C:\Users\Aditya Sawant\AppData\Local\Programs\Python\Python312\python.exe"
) else (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3.12"
    ) else (
        set "PYTHON_EXE=python"
    )
)

"%PYTHON_EXE%" "%~dp0..\engine\edith_brain.py"
pause
