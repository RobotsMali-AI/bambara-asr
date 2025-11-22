#!/usr/bin/env python3
# clean_manifest_remove_question_brackets.py
import argparse, json, sys, re
from pathlib import Path

PAT = re.compile(r"\<T>")           # literal <T>
WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = PAT.sub("", s)
    s = WS.sub(" ", s).strip()
    return s

def load_manifest(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if txt.startswith("["):
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError("JSON starts with '[' but isn't a list of objects.")
        return data, "json_array"
    # assume JSONL
    items = []
    for ln, line in enumerate(txt.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {ln}: {e}") from e
    return items, "jsonl"

def save_manifest(items, path: Path, mode: str):
    if mode == "json_array":
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Remove all occurrences of '[?]' from a NeMo ASR manifest.")
    ap.add_argument("input", help="Path to input manifest (JSONL or JSON array).")
    ap.add_argument("-o", "--output", help="Path to write cleaned manifest. Defaults to <input>.clean.jsonl/json.")
    ap.add_argument("--drop-empty", action="store_true", help="Drop entries whose text becomes empty after cleaning.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    items, mode = load_manifest(in_path)

    total = len(items)
    changed = 0
    removed = 0
    cleaned = []

    for obj in items:
        text = obj.get("text", "")
        new_text = clean_text(text)
        if new_text != text:
            changed += 1
        if args.drop_empty and new_text == "":
            removed += 1
            continue
        obj = dict(obj)
        obj["text"] = new_text
        cleaned.append(obj)

    out_path = Path(args.output) if args.output else (
        in_path.with_suffix(".clean.json") if mode == "json_array" else in_path.with_suffix(".clean.jsonl")
    )
    save_manifest(cleaned, out_path, mode)

    kept = len(cleaned)
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Total items:   {total}")
    print(f"Changed items: {changed}")
    if args.drop_empty:
        print(f"Dropped empty: {removed}")
    print(f"Kept:          {kept}")

if __name__ == "__main__":
    main()
