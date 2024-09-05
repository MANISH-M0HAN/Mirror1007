#!/bin/bash

set -e

# Update package list and install prerequisites
sudo apt update
sudo apt install -y software-properties-common
sleep 1

sudo apt install python3
sleep 1
REQUIREMENTS_FILE="requirements.txt"

# Install required Python packages
python3 -m pip install --break-system-packages -r "$REQUIREMENTS_FILE"

echo "Required packages installed from $REQUIREMENTS_FILE"
sleep 1

