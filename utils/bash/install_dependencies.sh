#!/usr/bin/env bash

### Copyright 2025 RobotsMali AI4D Lab.
### Licensed under the MIT License (https://opensource.org/licenses/MIT)

# Usage: ./install_deps.sh <requirements.txt>
# Example: ./install_deps.sh requirements.txt

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <requirements.txt> <venv_path>"
    exit 1
fi

REQ_FILE="$1"
VENV_PATH="$2"

if [ ! -f "$REQ_FILE" ]; then
    echo "Error: Requirements file '$REQ_FILE' not found."
    exit 1
fi

# Install Python 3.10 (ubuntu)
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Create a virtual environment
python3.10 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# OS packages (Ubuntu)
if ! sudo apt-get install -y libsndfile1 ffmpeg; then
    echo "Error installing OS packages. Check permissions."
    exit 1
fi

# Install Python dependencies from requirements.txt
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo "All dependencies installed successfully."
