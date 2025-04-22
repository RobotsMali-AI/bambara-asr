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
import json
import argparse

def derive_labeler(file_upload: str) -> str:
    """
    Map upload filename to labeler identifier.
    """
    base = os.path.basename(file_upload)
    # Expected patterns: '<name>-manifest.json', possibly prefixed by uuid-
    if 'djeliaV1-manifest.json' in base:
        return 'djeliaV1'
    if 'soloni-manifest.json' in base:
        return 'soloni'
    if 'shuffled-manifest.json' in base:
        return 'random'
    if 'benchmark-manifest.json' in base:
        return 'human'
    # fallback: strip suffix
    if base.endswith('-manifest.json'):
        return base.replace('-manifest.json', '')
    return base


def extract_score(record: dict) -> float:
    """
    Extract the human-assigned score from annotations; fallback to agreement if needed.
    """
    annotations = record.get('annotations', [])
    if annotations:
        # take first annotation's result value
        ann = annotations[0]
        results = ann.get('result', [])
        if results:
            return float(results[0].get('value', {}).get('number', 0.0))
    # fallback: use agreement
    return float(record.get('agreement', 0.0))


def main():
    """
    Script to convert messy HumanSignal JSON exports into a clean JSONL manifest.

    Each output line has:
    - audio_filepath: relative path under 'audios/'
    - duration: float
    - text: string transcription
    - score: float assigned by human annotator
    - labeler: identifier derived from source filename

    Example usage:
    python convert_annotation_to_manifest.py \
        --input messy.json --output manifest.jsonl
    """
    parser = argparse.ArgumentParser(description="Convert HumanSignal JSON to JSONL manifest.")
    parser.add_argument('--input', '-i', required=True,
                        help='Path to messy JSON export file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output JSONL manifest')
    args = parser.parse_args()

    # Load JSON (could be array or top-level dict)
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # if dict with 'results' key
    if isinstance(data, dict) and 'results' in data:
        records = data['results']
    else:
        records = data

    with open(args.output, 'w', encoding='utf-8') as out:
        for rec in records:
            # audio URL or path
            audio_src = rec.get('data', {}).get('audio_filepath', '')
            fname = os.path.basename(audio_src)
            audio_rel = os.path.join('audios', fname)

            duration = rec.get('data', {}).get('duration', None)
            text = rec.get('data', {}).get('text', '').strip()

            score = extract_score(rec)

            labeler = derive_labeler(rec.get('file_upload', ''))

            out_rec = {
                'audio_filepath': audio_rel,
                'duration': duration,
                'text': text,
                'score': score,
                'labeler': labeler,
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + '\n')

    print(f"Wrote manifest with {len(records)} entries to {args.output}")

if __name__ == '__main__':
    main()
