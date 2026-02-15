#!/bin/bash
# InfraScope Quick Start Script (macOS/Linux)

echo ""
echo "================================================"
echo "  InfraScope - Quick Start Setup"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/5] Python found:"
python3 --version

# Navigate to backend
cd backend

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[2/5] Virtual environment already exists"
fi

# Activate venv
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[4/5] Installing dependencies (this may take a few minutes)..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Start backend server
echo "[5/5] Starting InfraScope Backend Server..."
echo ""
echo "================================================"
echo "  Server starting on http://localhost:5000"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""

python app.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start server"
    exit 1
fi
