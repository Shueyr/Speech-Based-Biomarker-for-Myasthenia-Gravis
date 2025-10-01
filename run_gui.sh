#!/bin/bash
# Launcher script for MG Classifier GUI

echo "Starting MG Classifier GUI..."
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Using Python: $PYTHON_CMD"
echo "Location: $(which $PYTHON_CMD)"
echo "Version: $($PYTHON_CMD --version)"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the GUI
$PYTHON_CMD mg_classifier_gui.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Error: Application exited with an error"
    echo "=========================================="
    exit 1
fi

