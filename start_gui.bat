@echo off
REM BakkesMod RAG - GUI Startup Script (Windows)
REM This script launches the web-based GUI interface

echo ======================================
echo  BakkesMod RAG - GUI Launcher
echo ======================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your API keys:
    echo   copy .env.example .env
    echo   # Then edit .env and add your keys
    echo.
    pause
)

REM Launch GUI
echo.
echo ======================================
echo  Starting BakkesMod RAG GUI...
echo ======================================
echo.
echo The GUI will open in your browser at:
echo   http://localhost:7860
echo.
echo Press Ctrl+C to stop the server.
echo.

python rag_gui.py
pause
