"""
Copyright 2025 RobotsMali AI4D Lab.

Licensed under the MIT License; you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:

https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.
"""
from typing import Any
import json
from omegaconf import DictConfig, OmegaConf
from pydub import AudioSegment

def _omegaconf_to_container(obj: Any) -> Any:
    """Recursively convert DictConfig objects into plain dicts/lists so they're JSON-serializable."""
    if isinstance(obj, DictConfig):
        # convert DictConfig -> dict
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, list):
        return [_omegaconf_to_container(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _omegaconf_to_container(v) for k, v in obj.items()}
    return obj

def convert_to_mono(file_path):
    """Convert an audio file to mono if it has multiple channels."""
    audio = AudioSegment.from_file(file_path)
    if audio.channels > 1:
        audio = audio.set_channels(1)
        audio.export(file_path, format="wav")

def check_and_convert_audio_channels(manifest_path):
    """Check the number of channels in audio files and convert to mono if necessary."""
    with open(manifest_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        audio_path = json.loads(line)['audio_filepath']
        audio = AudioSegment.from_file(audio_path)
        if audio.channels > 1:
            convert_to_mono(audio_path)
    print(f"All stereo audios in {manifest_path} have been converted to mono")
