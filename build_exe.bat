@echo off
REM Build BakkesMod RAG GUI Windows Executable
REM ===========================================

echo.
echo ========================================
echo   BakkesMod RAG GUI - Executable Builder
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Please ensure Python 3.8+ is installed
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

REM Run build script
echo.
echo Building executable...
python build_exe.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo   Build FAILED
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Build SUCCESSFUL!
echo ========================================
echo.
echo Your executable is in: dist\BakkesMod_RAG_GUI\
echo.

pause
