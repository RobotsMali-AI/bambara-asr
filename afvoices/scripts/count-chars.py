#!/usr/bin/env python3
# unique_chars_in_manifest.py
import json
import argparse
from pathlib import Path
import unicodedata

def main():
    parser = argparse.ArgumentParser(description="Count unique characters in a manifest.jsonl")
    parser.add_argument("manifest", help="Path to the manifest JSONL file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    unique_chars = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Invalid JSON on line {line_num}: {e}")
                continue
            text = obj.get("text", "")
            # Normalize to NFC to avoid combining accents counting separately
            text = unicodedata.normalize("NFC", text)
            unique_chars.update(text)

    print(f"Number of unique characters: {len(unique_chars)}")
    print("Characters:")
    for ch in sorted(unique_chars):
        code = f"U+{ord(ch):04X}"
        name = unicodedata.name(ch, "UNKNOWN")
        print(f"  {repr(ch)} ({code} {name})")

if __name__ == "__main__":
    main()
