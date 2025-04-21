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


# Set default values
download_dir="."
tokenizer_dir="./bam-tokenizer"
manifest_dir="./bam-asr-all/manifests"

# Function to display help message
display_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "\nOptions:"
  echo "  --vocab_size=<size>           Vocabulary size for tokenizer"
  echo "  --spe_type=<type>             Type of SentencePiece tokenizer"
  echo "  --download_dir=<path>         Directory to download the dataset (default: .)"
  echo "  --tokenizer_dir=<path>        Directory for tokenizer output (default: ./bam-tokenizer)"
  echo "  --create_tokenizer_script_path=<path> Path to the script used to create the tokenizer"
  echo "  -h, --help                    Show this help message and exit"
  exit 0
}


# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --vocab_size=*) vocab_size="${1#*=}" ;;
    --spe_type=*) tokenizer_type="${1#*=}" ;;
    --download_dir=*) download_dir="${1#*=}" ;;
    --tokenizer_dir=*) tokenizer_dir="${1#*=}" ;;
    --create_tokenizer_script_path=*) create_tokenizer_script_path="${1#*=}" ;;
    -h|--help) display_help ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# Ensure the download directory exists
mkdir -p "$download_dir"
cd "$download_dir"

# Define file paths
audio_tar="bam-asr-all-1.0.0-audios.tar.gz"
manifest_tar="bam-asr-all-1.0.1-manifests.tar.gz"

# Download the audios if not already downloaded
if [ ! -f "$audio_tar" ]; then
  echo "Downloading audios..."
  wget -q https://huggingface.co/datasets/RobotsMali/bam-asr-all/resolve/archives/audio-archives/$audio_tar
else
  echo "Audios archive already exists. Skipping download."
fi

# Download the manifests if not already downloaded
if [ ! -f "$manifest_tar" ]; then
  echo "Downloading manifests..."
  wget -q https://huggingface.co/datasets/RobotsMali/bam-asr-all/resolve/archives/manifests-archives/$manifest_tar
else
  echo "Manifests archive already exists. Skipping download."
fi

# Extract audios if not already extracted
echo "Extracting audios..."
tar -xvzf "$audio_tar"

# Extract manifests if not already extracted
echo "Extracting manifests..."
tar -xvzf "$manifest_tar"

# Run the tokenizer script
python $create_tokenizer_script_path \
  --manifest="$manifest_dir/train-manifest.json" \
  --vocab_size="$vocab_size" \
  --data_root="$tokenizer_dir" \
  --tokenizer="spe" \
  --spe_type="$tokenizer_type" \
  --spe_character_coverage=1.0
