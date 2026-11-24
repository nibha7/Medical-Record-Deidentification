
from pathlib import Path
import json
import random
import joblib

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from common import (
    load_jsonl,
    bio_from_entities,
    tokenize_with_offsets,
    LABELS
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

# BIO labels for CNN
BIO_LABELS = ["O"] + [
    prefix + label
    for label in LABELS[1:]
    for prefix in ["B-", "I-"]
]

BIO_LABEL2ID = {
    label: i for i, label in enumerate(BIO_LABELS)
}


# --------------------------------------------------
# DECISION TREE AND RANDOM FOREST
# --------------------------------------------------

def make_token_rows(records):

    rows = []

    for record in records:

        tokens, labels = bio_from_entities(
            record["text"],
            record["entities"]
        )

        words = [token[0] for token in tokens]

        for i, token in enumerate(tokens):
            rows.append((words, i, labels[i]))

    return rows


def shape(word):

    return {
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "has_digit": any(c.isdigit() for c in word),
        "has_at": "@" in word,
        "has_slash": "/" in word,
        "len": min(len(word), 20),
        "prefix2": word[:2].lower(),
        "prefix3": word[:3].lower(),
        "suffix2": word[-2:].lower(),
        "suffix3": word[-3:].lower()
    }


def feature_dict(words, i):

    word = words[i]

    d = shape(word)

    d["word"] = word.lower()

    d["prev1"] = (
        words[i-1].lower() if i > 0 else "<START>"
    )

    d["next1"] = (
        words[i+1].lower()
        if i+1 < len(words) else "<END>"
    )

    d["prev2"] = (
        words[i-2].lower() if i > 1 else "<START>"
    )

    d["next2"] = (
        words[i+2].lower()
        if i+2 < len(words) else "<END>"
    )

    return d


def train_tree_models(train_records, val_records):

    train_rows = make_token_rows(train_records)

    X_dict = [
        feature_dict(words, i)
        for words, i, label in train_rows
    ]

    # Keep the original entity-class labels for DT and RF
    y = [
        label.split("-", 1)[-1] if "-" in label else label
        for words, i, label in train_rows
    ]

    vectorizer = DictVectorizer(sparse=True)

    X = vectorizer.fit_transform(X_dict)

    dt = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=2,
        random_state=SEED,
        class_weight="balanced"
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=24,
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    dt.fit(X, y)
    rf.fit(X, y)

    joblib.dump(
        vectorizer,
        MODEL_DIR / "feature_vectorizer.joblib"
    )

    joblib.dump(
        dt,
        MODEL_DIR / "decision_tree.joblib"
    )

    joblib.dump(
        rf,
        MODEL_DIR / "random_forest.joblib"
    )

    print("Decision Tree and Random Forest trained.")


# --------------------------------------------------
# CNN DATASET
# --------------------------------------------------

class WindowDataset(Dataset):

    def __init__(
        self,
        records,
        vocab,
        label2id,
        window=5
    ):

        self.samples = []
        self.window = window

        half = window // 2

        for record in records:

            tokens, labels = bio_from_entities(
                record["text"],
                record["entities"]
            )

            words = [
                token[0].lower()
                for token in tokens
            ]

            ids = [
                vocab.get(word, vocab["<UNK>"])
                for word in words
            ]

            # Preserve B- and I- prefixes
            label_ids = [
                label2id.get(label, 0)
                for label in labels
            ]

            for i in range(len(ids)):

                context = []

                for j in range(i-half, i+half+1):

                    if 0 <= j < len(ids):
                        context.append(ids[j])
                    else:
                        context.append(vocab["<PAD>"])

                self.samples.append(
                    (context, label_ids[i])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x, y = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


# --------------------------------------------------
# CNN MODEL
# --------------------------------------------------

class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_classes,
        emb=64,
        channels=96
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            emb,
            padding_idx=0
        )

        self.conv = nn.Conv1d(
            emb,
            channels,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(
            channels,
            n_classes
        )

    def forward(self, x):

        z = self.embedding(x).transpose(1, 2)

        z = torch.relu(self.conv(z))

        z = self.pool(z).squeeze(-1)

        return self.fc(z)


# --------------------------------------------------
# BUILD VOCABULARY
# --------------------------------------------------

def build_vocab(records):

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for record in records:

        for token, start, end in tokenize_with_offsets(
            record["text"]
        ):

            word = token.lower()

            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


# --------------------------------------------------
# TRAIN CNN WITH BIO LABELS
# --------------------------------------------------

def train_cnn(train_records, val_records):

    print("\nStarting corrected CNN training...")

    vocab = build_vocab(train_records)

    # Save new CNN vocabulary separately
    joblib.dump(
        vocab,
        MODEL_DIR / "cnn_bio_vocab.joblib"
    )

    with open(
        MODEL_DIR / "cnn_bio_labels.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            BIO_LABELS,
            f,
            indent=2
        )

    dataset = WindowDataset(
        train_records,
        vocab,
        BIO_LABEL2ID,
        window=5
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    model = TextCNN(
        len(vocab),
        len(BIO_LABELS)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(1, 11):

        total_loss = 0.0

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            output = model(x)

            loss = loss_function(output, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            "CNN epoch {}/10 loss={:.4f}".format(
                epoch,
                average_loss
            )
        )

    # Save corrected CNN separately
    torch.save(
        model.state_dict(),
        MODEL_DIR / "cnn_bio_model.pt"
    )

    print("\nCorrected BIO-label CNN trained successfully.")

    print("Saved:")
    print("models/cnn_bio_model.pt")
    print("models/cnn_bio_vocab.joblib")
    print("models/cnn_bio_labels.json")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    train_records = load_jsonl("data/train.jsonl")

    val_records = load_jsonl("data/val.jsonl")

    print("Training records:", len(train_records))
    print("Validation records:", len(val_records))

    train_tree_models(
        train_records,
        val_records
    )

    train_cnn(
        train_records,
        val_records
    )

    print("\nAll base models trained successfully.")
