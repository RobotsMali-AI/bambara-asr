import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Create train and test split from downloaded manifest.")
    parser.add_argument("manifest", type=str, help="Path to the downloaded manifest file.")
    parser.add_argument("--test-size", type=float, default=3., help="Proportion of the dataset to separate into test split (in hours).")
    parser.add_argument("--train-manifest", type=str, default="train-manifest.jsonl", help="Output path for the training manifest.")
    parser.add_argument("--test-manifest", type=str, default="test-manifest.jsonl", help="Output path for the test manifest.")
    args = parser.parse_args()
    
    entries = [json.loads(line) for line in open(args.manifest, "r", encoding="utf-8")]
    total_duration = sum(entry.get("duration", 0) for entry in entries)
    print(f"Total dataset duration: {total_duration/3600:.2f} hours\nPreparing train-test split; test size: {args.test_size} hours")
    random.shuffle(entries)
    test_entries = []
    accumulated_duration = 0.0
    for index, entry in enumerate(entries):
        if accumulated_duration < args.test_size * 3600:
            test_entries.append(entries.pop(index))
            accumulated_duration += entry.get("duration", 0)
        else:
            break
    train_entries = entries
    with open(args.train_manifest, "w", encoding="utf-8") as f:
        for e in train_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(args.test_manifest, "w", encoding="utf-8") as f:
        for e in test_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("✅ Train-test split complete:")
    print(f"   Training set: {len(train_entries)} entries, {sum(e.get('duration', 0) for e in train_entries)/3600:.2f} hours")
    print(f"   Test set: {len(test_entries)} entries, {sum(e.get('duration', 0) for e in test_entries)/3600:.2f} hours")

if __name__ == "__main__":
    main()
