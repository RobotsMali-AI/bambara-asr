#!/bin/bash

### Copyright 2025 RobotsMali AI4D Lab.

### Licensed under the MIT License; you may not use this file except in compliance with the License.  
### You may obtain a copy of the License at:

### https://opensource.org/licenses/MIT

### Unless required by applicable law or agreed to in writing, software  
### distributed under the License is distributed on an "AS IS" BASIS,  
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
### See the License for the specific language governing permissions and  
### limitations under the License.

# This script installs the dependencies for the fine-tuning scripts in this repository.
# If you want to install them in a specific environment, ensure you are in the right environment before running this script.

# First of all, install pip

sudo apt install python3-pip

# OS packages (Assuming you are using Ubuntu)
if ! apt-get install -y libsndfile1 ffmpeg; then
    echo "Insufficient permissions or error occurred. Trying with sudo... "
    echo "Ensure you have sudo privileges. You may be prompted for your password..."
    sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
fi

# Install NeMo and its dependencies
# Note that installing NeMo through pip will install PyTorch Lightning and other dependencies
pip install Cython packaging
pip install nemo_toolkit['asr']
pip install megatron-core
# Install miscellaneous dependencies
pip install pydub
pip install wandb