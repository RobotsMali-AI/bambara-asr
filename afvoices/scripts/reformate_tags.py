import json
import re
import sys
import os

def load_manifest(path: str):
    """Load JSON or JSONL manifest into a list of dicts."""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format (must be .json or .jsonl)")

def write_manifest(path: str, data):
    """Write list of dicts back to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def normalize_tags(text: str) -> str:
    """Replace [tag] with its uppercase form (no brackets)."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\[(.*?)\]", lambda m: m.group(1).strip().upper(), text)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input_manifest> <output_manifest>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    manifest = load_manifest(in_path)

    normalized = []
    for entry in manifest:
        new_entry = dict(entry)
        if "text" in new_entry and isinstance(new_entry["text"], str):
            new_entry["text"] = normalize_tags(new_entry["text"]).replace("?", "UNINTELLIGIBLE")
        normalized.append(new_entry)

    write_manifest(out_path, normalized)
    print(f"✅ Normalized manifest saved to: {out_path}")
    print(f"Total entries processed: {len(normalized)}")

if __name__ == "__main__":
    main()
