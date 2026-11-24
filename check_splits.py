
import json
from pathlib import Path

files = {
    "train": Path("data/train.jsonl"),
    "validation": Path("data/val.jsonl"),
    "test": Path("data/test.jsonl"),
}

data = {}

for name, path in files.items():
    if not path.exists():
        print(f"MISSING: {path}")
        continue

    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    normalized = [
        " ".join(record["text"].lower().split())
        for record in records
    ]

    data[name] = set(normalized)

    print(f"{name}: {len(records)} records")
    print(f"{name}: {len(data[name])} unique texts")

for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    if a in data and b in data:
        overlap = data[a] & data[b]
        print(f"\n{a} vs {b}: {len(overlap)} duplicate texts")
