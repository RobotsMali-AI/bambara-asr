import pandas as pd
import json
import argparse

def load_manifest(path: str) -> list[dict]:
    if path.endswith("json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.endswith("jsonl"):
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")
    return data

def main():
    parser = argparse.ArgumentParser(description="Describe duration statistics from a manifest file.")
    parser.add_argument("input_path", type=str, help="Path to the manifest file (JSON or JSONL format).")
    args = parser.parse_args()
    print("Loading manifest from:", args.input_path)
    data = load_manifest(args.input_path)
    df = pd.DataFrame(data)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    print("=== Duration Summary ===")
    print(df['duration'].describe())
    print("Total duration (Hours):", df['duration'].sum() / 3600)
    print("=== End of Summary ===")

if __name__ == "__main__":
    main()

