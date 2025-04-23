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
import os
import argparse
import json
import shutil
from tqdm import tqdm
from datasets import load_dataset, Audio, DatasetDict


def main():
    """
    Convert a Hugging Face dataset into a NeMo ASR dataset layout.

    Usage:
    python hf_to_nemo_asr.py \
        --repo_id <hf_repo> \
        --subset <subset_name> \
        --save_dir <output_dir>

    This script will:
    - Load the Hugging Face dataset (`repo_id`, `subset`).
    - For each split (train/test/validation), create:
        save_dir/audios/<split>/
        save_dir/<split>-manifest.jsonl
    - Copy each example's audio file into the split's audio dir,
        preserving basename when present, otherwise naming as <split>_<i>.wav.
    - Write a JSONL manifest per split listing:
        { "audio_filepath": <relative_path>, <all other columns> }
    - Print paths to generated manifests.
    """
    parser = argparse.ArgumentParser(description="Convert HF dataset to NeMo ASR format")
    parser.add_argument('--repo_id', type=str, required=True, help='Hugging Face dataset repo id')
    parser.add_argument('--subset', type=str, default=None, help='Subset or config name')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save NeMo dataset')
    args = parser.parse_args()

    repo_id = args.repo_id
    subset = args.subset
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load HF dataset
    print(f"Loading dataset {repo_id}{' config='+subset if subset else ''}...")
    if subset:
        ds = load_dataset(repo_id, subset)
    else:
        ds = load_dataset(repo_id)
    assert isinstance(ds, DatasetDict), "Expected a DatasetDict with splits"

    manifests = {}
    # Prepare audio base dir
    audio_base = os.path.join(save_dir, 'audios')
    os.makedirs(audio_base, exist_ok=True)

    for split, dataset in ds.items():
        split_dir = os.path.join(audio_base, split)
        os.makedirs(split_dir, exist_ok=True)
        manifest_path = os.path.join(save_dir, f'{split}-manifest.jsonl')
        print(f"Processing split '{split}' ({len(dataset)} samples)")

        with open(manifest_path, 'w', encoding='utf-8') as mf:
            for i, example in tqdm(enumerate(dataset), total=len(dataset)):
                # Determine audio source path
                audio_field = None
                for col, feat in dataset.features.items():
                    if isinstance(feat, Audio):
                        audio_field = col
                        break
                if audio_field is None:
                    raise ValueError("No Audio feature found in dataset")
                audio_info = example[audio_field]
                # audio_info may be dict with 'path'
                src_path = audio_info.get('path', None) if isinstance(audio_info, dict) else None
                if src_path:
                    fname = os.path.basename(src_path)
                    # Ensure .wav extension
                    if not fname.endswith('.wav'):
                        fname = os.path.splitext(fname)[0] + '.wav'
                else:
                    fname = f"{split}_{i}.wav"
                dest_rel = os.path.join('audios', split, fname)
                dest_path = os.path.join(save_dir, dest_rel)

                # Copy if path exists
                if src_path and os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                else:
                    # try streaming download
                    try:
                        # download via dataset.load_audio
                        arr = dataset[i][audio_field]['array']
                        sample_rate = dataset[i][audio_field]['sampling_rate']
                        import soundfile as sf
                        sf.write(dest_path, arr, sample_rate)
                    except Exception:
                        raise RuntimeError(f"Cannot retrieve audio for example {i} in split {split}")

                # Build manifest entry: audio_filepath + other fields
                entry = {'audio_filepath': dest_path}
                # copy other fields except audio_field
                for k, v in example.items():
                    if k == audio_field:
                        # include duration if present
                        if isinstance(v, dict) and 'duration' in v:
                            entry['duration'] = v['duration']
                        continue
                    entry[k] = v
                mf.write(json.dumps(entry, ensure_ascii=False) + '\n')
        manifests[split] = manifest_path

    print("Generated manifests:")
    for split, path in manifests.items():
        print(f"  {split}: {path}")

if __name__ == '__main__':
    main()
