#!/bin/bash

set -e

if ! command -v brew &> /dev/null; then
  echo "Homebrew is not installed. Please install Homebrew before running this script."
  exit 1
fi

brew install python3
if ! command -v python3 &> /dev/null; then
  echo "Python 3 installation failed. Please install Python 3 manually."
  exit 1
fi
echo "Python 3 installed successfully."

REQUIREMENTS_FILE="requirements.txt"

sleep 1

python3 -m pip install --break-system-packages -r "$REQUIREMENTS_FILE"
echo "Required packages installed from $REQUIREMENTS_FILE"
