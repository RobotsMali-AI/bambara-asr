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

# Function to check .wav file count
check_wav_count() {
  dir_path="$1"
  dataset_size="$2"
  count=$(find "$dir_path" -type f -name "*.wav" | wc -l)
  if [ "$count" -eq "$dataset_size" ]; then
    echo "✔ $dir_path contains the correct number of .wav audio samples: $count"
    return 0
  else
    echo "❌ ERROR: $dir_path contains $count .wav files instead of $dataset_size"
    return 1
  fi
}

# Function to check if the tokenizer directory contains the required files
check_tokenizer() {
  tokenizer_dir="$1"
  required_files=("tokenizer.model" "tokenizer.vocab" "vocab.txt")
  missing_files=()

  for file in "${required_files[@]}"; do
    if [ ! -f "$tokenizer_dir/$file" ]; then
      missing_files+=("$file")
    fi
  done

  if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✔ Tokenizer is ready. All required files are present in $tokenizer_dir"
    return 0
  else
    echo "❌ ERROR: The following tokenizer files are missing in $tokenizer_dir: ${missing_files[*]}"
    return 1
  fi
}

# Function to check if dependencies are installed
check_dependencies() {
  system_dependencies=("pip" "ffmpeg" "libsndfile1" "python3")
  python_packages=("pydub" "nemo" "megatron" "wandb")

  missing_deps=()

  # Check system dependencies
  for dep in "${system_dependencies[@]}"; do
    if ! command -v "$dep" &>/dev/null && ! dpkg -l | grep -q "$dep"; then
      missing_deps+=("$dep (system)")
    fi
  done

  # Check Python dependencies
  for pkg in "${python_packages[@]}"; do
    if ! python3 -c "import $pkg" &>/dev/null; then
      missing_deps+=("$pkg (Python)")
    fi
  done

  # Report results
  if [ ${#missing_deps[@]} -eq 0 ]; then
    echo "✔ All dependencies are installed."
    return 0
  else
    echo "❌ ERROR: The following dependencies are missing:"
    printf "%s\n" "${missing_deps[@]}"
    echo "Run the installation script to resolve this issue."
    return 1
  fi
}

# Function to check wandb authentication
check_wandb() {
  if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ ERROR: No WANDB_API_KEY found in environment variables."
    echo "Please set it using: export WANDB_API_KEY=your_api_key"
    return 1
  fi
  
  wandb login --relogin "$WANDB_API_KEY"
  if [ $? -eq 0 ]; then
    echo "✔ Weights & Biases authentication successful."
    return 0
  else
    echo "❌ ERROR: Weights & Biases login failed. Ensure your API key is correct."
    return 1
  fi
}

# Function to display help message
display_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "\nOptions:"
  echo "  --dataset_size=<size>         Expected number of .wav files in the dataset"
  echo "  --dataset_dir=<path>          Path to the dataset directory"
  echo "  --tokenizer_dir=<path>        Path to the tokenizer directory"
  echo "  -h, --help                    Show this help message and exit"
  exit 0
}

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset_size=*) dataset_size="${1#*=}" ;;
    --dataset_dir=*) dataset_dir="${1#*=}" ;;
    --tokenizer_dir=*) tokenizer_dir="${1#*=}" ;;
    -h|--help) display_help ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

check_wav_count "$dataset_dir" "$dataset_size" || exit 1

if [ ! -d "$dataset_dir/manifests" ]; then
  echo "❌ ERROR: Manifests directory not found in $dataset_dir."
  exit 1
else
  echo "✔ Manifests directory found in $dataset_dir."
fi

check_tokenizer "$tokenizer_dir" || exit 1
check_dependencies || exit 1
check_wandb || exit 1

echo "✔ All checks passed. System is ready for training."
