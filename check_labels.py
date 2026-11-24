
from collections import Counter
from common import load_jsonl, bio_from_entities, LABELS

records = load_jsonl("data/test.jsonl")

print("Number of test records:", len(records))
print("Model labels:", LABELS)

label_counts = Counter()
entity_counts = Counter()

for record in records:
    _, bio_labels = bio_from_entities(
        record["text"],
        record["entities"]
    )

    for label in bio_labels:
        # Convert B-NAME and I-NAME to NAME, for example.
        label = label.split("-", 1)[-1]
        label_counts[label] += 1

    for entity in record["entities"]:
        if isinstance(entity, dict):
            entity_label = entity.get(
                "label",
                entity.get("type", "UNKNOWN")
            )
            entity_counts[entity_label] += 1

print("\nGold token-label counts:")
for label, count in label_counts.most_common():
    print(f"{label}: {count}")

print("\nEntity counts from test records:")
for label, count in entity_counts.most_common():
    print(f"{label}: {count}")

print("\nTotal labelled tokens:", sum(label_counts.values()))
print("Total PHI tokens:", sum(
    count for label, count in label_counts.items()
    if label != "O"
))
