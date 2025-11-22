#!/usr/bin/env bash

### Copyright 2025 RobotsMali AI4D Lab.
### Licensed under the MIT License (https://opensource.org/licenses/MIT)

# Usage: ./install_deps.sh <requirements.txt>
# Example: ./install_deps.sh requirements.txt

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <requirements.txt>"
    exit 1
fi

REQ_FILE="$1"

if [ ! -f "$REQ_FILE" ]; then
    echo "Error: Requirements file '$REQ_FILE' not found."
    exit 1
fi

# Install pip if not already installed
sudo apt update
sudo apt install -y python3-pip

# OS packages (Ubuntu)
if ! sudo apt-get install -y libsndfile1 ffmpeg; then
    echo "Error installing OS packages. Check permissions."
    exit 1
fi

# Install Python dependencies from requirements.txt
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo "All dependencies installed successfully."
