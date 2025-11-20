#!/usr/bin/env python3
# replace_tags.py
import argparse, json, re, sys
from pathlib import Path

TAG_RE = re.compile(r"<[^>\n]+>")  # anything like <...> on a single line

def replace_tags_in_text(text: str) -> tuple[str, int]:
    """Replace all <...> chunks by [?]. Returns (new_text, num_replacements)."""
    count = 0
    def _sub(_):
        nonlocal count
        count += 1
        return "<T>"
    return TAG_RE.sub(_sub, text), count

def main():
    ap = argparse.ArgumentParser(description="Replace <TAGS> in JSONL manifest with [?].")
    ap.add_argument("--input", required=True, help="Path to input manifest.jsonl")
    ap.add_argument("--output", required=True, help="Path to write the new manifest.jsonl")
    ap.add_argument("--field", default="text", help="JSON field to process (default: text)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_lines = n_modified = n_tags = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON on line {n_lines}: {e}", file=sys.stderr)
                continue

            if args.field in obj and isinstance(obj[args.field], str):
                new_text, k = replace_tags_in_text(obj[args.field])
                if k > 0:
                    obj[args.field] = new_text
                    n_modified += 1
                    n_tags += k

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. Lines: {n_lines} | Modified lines: {n_modified} | Tags replaced: {n_tags}")
    print(f"Output -> {out_path}")

if __name__ == "__main__":
    main()
