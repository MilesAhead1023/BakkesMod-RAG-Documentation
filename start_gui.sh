#!/bin/bash
# BakkesMod RAG - GUI Startup Script
# This script launches the web-based GUI interface

echo "======================================"
echo " BakkesMod RAG - GUI Launcher"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env and add your keys"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Launch GUI
echo ""
echo "======================================"
echo " Starting BakkesMod RAG GUI..."
echo "======================================"
echo ""
echo "The GUI will open in your browser at:"
echo "  http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

python rag_gui.py
