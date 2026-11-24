import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from common import LABELS, load_jsonl, tokenize_with_offsets, bio_from_entities

SEED = 42
TRAIN_PATH = Path("data/train.jsonl")
VAL_PATH = Path("data/val.jsonl")
MODEL_PATH = Path("models/bilstm_crf/model.pt")
VOCAB_PATH = Path("models/bilstm_crf/vocab.json")
LABELS_PATH = Path("models/bilstm_crf/labels.json")

EMBED_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIO_LABELS = ["O"]
for entity in LABELS:
    if entity != "O":
        BIO_LABELS.extend(["B-" + entity, "I-" + entity])
LABEL2ID = {x: i for i, x in enumerate(BIO_LABELS)}
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def records_to_sequences(records):
    output = []
    for record in records:
        tokens, bio = bio_from_entities(record["text"], record.get("entities", []))
        words = [token[0].lower() for token in tokens]
        # Normalize any unexpected label to O rather than crashing.
        bio = [tag if tag in LABEL2ID else "O" for tag in bio]
        output.append((words, [LABEL2ID[tag] for tag in bio]))
    return output


def build_vocab(sequences):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for words, _ in sequences:
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


class SequenceDataset(Dataset):
    def __init__(self, sequences, vocab):
        self.items = []
        for words, labels in sequences:
            ids = [vocab.get(word, vocab[UNK_TOKEN]) for word in words]
            self.items.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_batch(batch):
    tokens, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in tokens], dtype=torch.long)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    mask = tokens != 0
    return tokens, labels, mask, lengths


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

    def _log_partition(self, emissions, mask):
        # emissions: [batch, seq, tags]
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def _gold_score(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags]
        score += emissions[torch.arange(batch_size, device=tags.device), 0, first_tags]

        for t in range(1, seq_len):
            active = mask[:, t]
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]
            step = self.transitions[prev_tags, curr_tags]
            step += emissions[torch.arange(batch_size, device=tags.device), t, curr_tags]
            score += step * active

        last_indices = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size, device=tags.device), last_indices]
        score += self.end_transitions[last_tags]
        return score

    def neg_log_likelihood(self, emissions, tags, mask):
        # Replace padded -100 tags with O for safe indexing; mask excludes them.
        safe_tags = tags.masked_fill(tags < 0, 0)
        return (self._log_partition(emissions, mask) -
                self._gold_score(emissions, safe_tags, mask)).mean()

    def decode(self, emissions, mask):
        results = []
        for b in range(emissions.size(0)):
            length = int(mask[b].sum().item())
            score = self.start_transitions + emissions[b, 0]
            history = []
            for t in range(1, length):
                candidates = score.unsqueeze(1) + self.transitions
                best_score, best_path = candidates.max(dim=0)
                score = best_score + emissions[b, t]
                history.append(best_path)
            score = score + self.end_transitions
            tag = int(score.argmax().item())
            path = [tag]
            for backpointer in reversed(history):
                tag = int(backpointer[tag].item())
                path.append(tag)
            results.append(list(reversed(path)))
        return results


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            EMBED_DIM, HIDDEN_DIM // 2,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(HIDDEN_DIM, num_tags)
        self.crf = LinearChainCRF(num_tags)

    def emissions(self, token_ids):
        embedded = self.embedding(token_ids)
        encoded, _ = self.lstm(embedded)
        return self.classifier(self.dropout(encoded))

    def loss(self, token_ids, tags, mask):
        return self.crf.neg_log_likelihood(self.emissions(token_ids), tags, mask)

    def predict(self, token_ids, mask):
        return self.crf.decode(self.emissions(token_ids), mask)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for tokens, labels, mask, _ in loader:
            tokens, labels, mask = tokens.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)
            paths = model.predict(tokens, mask)
            for row, path in enumerate(paths):
                gold = labels[row, :len(path)].tolist()
                for g, p in zip(gold, path):
                    if g >= 0:
                        total += 1
                        correct += int(g == p)
    return correct / total if total else 0.0


def main():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Expected data/train.jsonl and data/val.jsonl.")

    train_sequences = records_to_sequences(load_jsonl(TRAIN_PATH))
    val_sequences = records_to_sequences(load_jsonl(VAL_PATH))
    vocab = build_vocab(train_sequences)

    train_loader = DataLoader(
        SequenceDataset(train_sequences, vocab),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        SequenceDataset(val_sequences, vocab),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    model = BiLSTMCRF(len(vocab), len(BIO_LABELS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = -1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for tokens, labels, mask, _ in train_loader:
            tokens, labels, mask = tokens.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            loss = model.loss(tokens, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_accuracy = evaluate(model, val_loader)
        print(
            "Epoch {}/{} | loss={:.4f} | validation token accuracy={:.4f}".format(
                epoch + 1, EPOCHS,
                total_loss / max(len(train_loader), 1),
                val_accuracy
            )
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            with open(VOCAB_PATH, "w", encoding="utf-8") as f:
                json.dump(vocab, f)
            with open(LABELS_PATH, "w", encoding="utf-8") as f:
                json.dump(BIO_LABELS, f, indent=2)
            print("Saved best BiLSTM-CRF model.")

    print("Training complete. Best validation token accuracy:", round(best_val_accuracy, 4))
    print("Use exact entity-level test evaluation for final model comparisons.")


if __name__ == "__main__":
    main()
