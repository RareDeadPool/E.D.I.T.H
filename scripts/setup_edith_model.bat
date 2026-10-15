@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo       E.D.I.T.H. NEURAL NETWORK ENVIRONMENT SETUP        
echo =========================================================

:: Check Python 3.12 installation path
set "PYTHON_EXE="

if exist "C:\Users\Aditya Sawant\AppData\Local\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=C:\Users\Aditya Sawant\AppData\Local\Programs\Python\Python312\python.exe"
) else (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3.12"
    ) else (
        where python >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=python"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python 3.12 could not be found! Please verify Python installation.
    pause
    exit /b 1
)

echo [INFO] Using Python: %PYTHON_EXE%

:: Verify PyTorch
echo [INFO] Verifying PyTorch and dependencies...
"%PYTHON_EXE%" -c "import torch; print('  -> PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch not found. Installing requirements...
    "%PYTHON_EXE%" -m pip install -r "%~dp0..\requirements.txt"
)

:: Generate dataset
echo.
echo [INFO] Generating domain training and validation datasets...
"%PYTHON_EXE%" "%~dp0..\data\dataset_generator.py"

echo.
echo =========================================================
echo  Setup Completed Successfully!
echo  To train the ANN model, run:
echo    scripts\run_training.bat
echo =========================================================
echo.
pause
