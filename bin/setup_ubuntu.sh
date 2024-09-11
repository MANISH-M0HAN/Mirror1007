#!/bin/bash

set -e

# Update package list and install prerequisites
sudo apt update
sudo apt install -y software-properties-common

# Check if Python 3 is installed, if not, install it
if ! command -v python3 &> /dev/null; then
  sudo apt install -y python3 python3-pip
  if ! command -v python3 &> /dev/null; then
    echo "Python 3 installation failed. Please install Python 3 manually."
    exit 1
  fi
  echo "Python 3 installed successfully."
else
  echo "Python 3 is already installed."
fi

REQUIREMENTS_FILE="./python_scripts/requirements.txt"

# Install required Python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r "$REQUIREMENTS_FILE"
echo "Required packages installed from $REQUIREMENTS_FILE"