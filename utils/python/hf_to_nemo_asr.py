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
from pathlib import Path
from typing import Optional
import soundfile as sf
from datasets import load_dataset, Audio, DatasetDict, Features


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_audio_column(features: Features) -> Optional[str]:
    for name, feat in features.items():
        if isinstance(feat, Audio):
            return name
    return None


def write_manifest_line(fp, entry: dict):
    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def example_duration(example_audio_value) -> Optional[float]:
    # Works for both decode=False and decode=True
    if isinstance(example_audio_value, dict):
        # When decode=False, we often have {'path': str, 'bytes': None, 'sampling_rate': <maybe>, 'duration': <maybe>}
        dur = example_audio_value.get("duration")
        if isinstance(dur, (int, float)):
            return float(dur)
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert HF dataset to NeMo ASR format")
    parser.add_argument('--repo_id', type=str, required=True, help='Hugging Face dataset repo id')
    parser.add_argument('--subset', type=str, default=None, help='Subset/config name')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save NeMo dataset')
    parser.add_argument('--text-field', type=str, default='text', help='Field name for text transcription')
    parser.add_argument('--split', type=str, default=None, help='Only process a single split (train/test/validation)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--decode-fallback', action='store_true', help='Force per-example decode when source path missing')

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    audio_base = save_dir / 'audios'
    ensure_dir(audio_base)

    print(f"Loading dataset {args.repo_id}{' config='+args.subset if args.subset else ''}…")
    if args.subset:
        ds_any = load_dataset(args.repo_id, args.subset)
    else:
        ds_any = load_dataset(args.repo_id)

    if not isinstance(ds_any, DatasetDict):
        raise ValueError("Expected a DatasetDict with splits")

    manifests = {}

    for split, dataset in ds_any.items():
        if args.split and split != args.split:
            continue

        audio_col = pick_audio_column(dataset.features)
        if audio_col is None:
            raise ValueError(f"No Audio feature found in split '{split}'")

        split_dir = audio_base / split
        ensure_dir(split_dir)
        manifest_path = save_dir / f"original-{split}-manifest.jsonl"

        # Detect whether audio is currently decode=False on this split
        decode_flag = isinstance(dataset.features[audio_col], Audio) and dataset.features[audio_col].decode
        print(f"Split '{split}': {len(dataset)} rows | audio column='{audio_col}' | decode={decode_flag}")

        # We'll try to copy from local path when available. If missing and --decode-fallback, decode on the fly.
        # To avoid repeatedly recasting the whole dataset, we will cast only when we first need a decode.
        decoded_ds = None

        with open(manifest_path, 'w', encoding='utf-8') as mf:
            for i, ex in enumerate((dataset)):
                a = ex[audio_col]

                # Determine source path
                src_path = None
                if isinstance(a, dict):
                    src_path = a.get('path')

                # Choose destination file name
                if src_path:
                    fname = os.path.basename(src_path)
                    if not fname.lower().endswith('.wav'):
                        fname = os.path.splitext(fname)[0] + '.wav'
                else:
                    fname = f"{split}_{i}.wav"

                dest_rel = Path('audios') / split / fname
                dest_path = save_dir / dest_rel

                # Copy or decode
                need_copy = src_path and os.path.exists(src_path)
                if need_copy:
                    if args.overwrite or not dest_path.exists():
                        shutil.copy(src_path, dest_path)
                elif "array" in a and "sampling_rate" in a:
                    arr = a['array']
                    sr = a['sampling_rate']
                    ensure_dir(dest_path.parent)
                    if args.overwrite or not dest_path.exists():
                        sf.write(str(dest_path), arr, sr)
                else:
                    if not args.decode_fallback:
                        raise RuntimeError(
                            f"No local path for example {i} in split '{split}'. "
                            f"Re-run with --decode-fallback to decode and write audio."
                        )
                    # Lazy cast to decode=True only when needed
                    if decoded_ds is None:
                        print("[info] Falling back to per-example decode (cast decode=True) …")
                        decoded_ds = dataset.cast_column(audio_col, Audio(decode=True))
                    # Decode this item
                    item = decoded_ds[i][audio_col]
                    arr = item['array']
                    sr = item['sampling_rate']
                    ensure_dir(dest_path.parent)
                    if args.overwrite or not dest_path.exists():
                        sf.write(str(dest_path), arr, sr)
                    else:
                        print("file already exists. Adding entry to manifest")

                # Build manifest entry
                entry = {
                    'audio_filepath': str(dest_rel),
                }

                # duration: prefer explicit key in example, else audio dict duration
                dur = ex.get('duration')
                if dur is None:
                    dur = example_duration(a)
                if dur is not None:
                    entry['duration'] = float(dur)

                # text field
                if args.text_field in ex:
                    entry['text'] = ex[args.text_field]
                else:
                    # Keep going but warn once
                    if i == 0:
                        print(f"[warn] text field '{args.text_field}' not found; leaving out 'text' in manifest.")

                write_manifest_line(mf, entry)

        manifests[split] = str(manifest_path)
        print(f"[ok] Wrote {manifest_path}")

    print("Generated manifests:")
    for split, p in manifests.items():
        print(f"  {split}: {p}")


if __name__ == '__main__':
    main()
