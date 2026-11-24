
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_recall_fscore_support

from common import LABELS, load_jsonl

# ---------------- CONFIGURATION ----------------

SEED = 42
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_DIR = Path("models/bioclinicalbert")

TRAIN_PATH = Path("data/train.jsonl")
VAL_PATH = Path("data/val.jsonl")

MAX_LENGTH = 128
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-5

# Temporary subset for initial CPU testing
TRAIN_LIMIT = 1000
VAL_LIMIT = 200

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIO_LABELS = ["O"]

for entity in LABELS:
    if entity != "O":
        BIO_LABELS.extend(["B-" + entity, "I-" + entity])

LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ---------------- DATASET ----------------

class PHIDataset(Dataset):

    def __init__(self, records, tokenizer):

        self.items = []

        for record in records:

            text = record["text"]
            entities = record.get("entities", [])

            encoding = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_offsets_mapping=True
            )

            offsets = encoding.pop("offset_mapping")
            token_labels = []

            for start, end in offsets:

                if start == end:
                    token_labels.append(-100)
                    continue

                assigned = "O"

                for entity in entities:

                    es = int(entity["start"])
                    ee = int(entity["end"])
                    entity_label = entity["label"].upper()

                    if entity_label not in LABELS or entity_label == "O":
                        continue

                    if end > es and start < ee:

                        if start <= es < end:
                            assigned = "B-" + entity_label
                        else:
                            assigned = "I-" + entity_label

                        break

                token_labels.append(LABEL2ID[assigned])

            item = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in encoding.items()
            }

            item["labels"] = torch.tensor(
                token_labels, dtype=torch.long
            )

            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# ---------------- EVALUATION ----------------

def evaluate_token_level(model, loader):

    model.eval()

    gold = []
    pred = []

    with torch.no_grad():

        for batch in loader:

            labels = batch["labels"].to(DEVICE)

            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }

            logits = model(**inputs).logits
            predictions = logits.argmax(dim=-1)

            valid = labels != -100

            gold.extend(labels[valid].cpu().tolist())
            pred.extend(predictions[valid].cpu().tolist())

    entity_ids = list(range(1, len(BIO_LABELS)))

    p, r, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        labels=entity_ids,
        average="micro",
        zero_division=0
    )

    return p, r, f1


# ---------------- MAIN TRAINING ----------------

def main():

    if not TRAIN_PATH.exists() or not VAL_PATH.exists():

        raise FileNotFoundError(
            "Training or validation dataset not found."
        )

    print("Device:", DEVICE)

    print("Loading BioClinicalBERT:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(DEVICE)

    print("Loading datasets...")

    train_records = load_jsonl(TRAIN_PATH)
    val_records = load_jsonl(VAL_PATH)

    print("Original training records:", len(train_records))
    print("Original validation records:", len(val_records))

    train_records = train_records[:TRAIN_LIMIT]
    val_records = val_records[:VAL_LIMIT]

    print("Training subset:", len(train_records))
    print("Validation subset:", len(val_records))

    train_data = PHIDataset(train_records, tokenizer)
    val_data = PHIDataset(val_records, tokenizer)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0

    print("Starting training...")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for step, batch in enumerate(train_loader):

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            outputs.loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += outputs.loss.item()

            if (step + 1) % 50 == 0:
                print(
                    "Epoch {} | Batch {}/{} | Loss {:.4f}".format(
                        epoch + 1,
                        step + 1,
                        len(train_loader),
                        outputs.loss.item()
                    ),
                    flush=True
                )

        precision, recall, f1 = evaluate_token_level(
            model, val_loader
        )

        mean_loss = total_loss / max(len(train_loader), 1)

        print(
            "Epoch {} | Loss {:.4f} | Precision {:.4f} "
            "| Recall {:.4f} | F1 {:.4f}".format(
                epoch + 1,
                mean_loss,
                precision,
                recall,
                f1
            )
        )

        if f1 > best_f1:

            best_f1 = f1

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            with open(
                MODEL_DIR / "bio_labels.json",
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(BIO_LABELS, f, indent=2)

            print("Best model saved to:", MODEL_DIR)

    print("Training completed.")
    print("Best validation token F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
