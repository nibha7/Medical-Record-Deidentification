import json
import csv
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from common import (
    load_jsonl,
    tokenize_with_offsets,
    bio_from_entities
)

# ==================================================
# CONFIGURATION
# ==================================================

TEST_PATH = Path("data/test.jsonl")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULT_DIR.mkdir(exist_ok=True)


# ==================================================
# SPAN CONVERSION
# ==================================================

def spans_from_class_predictions(tokens, predictions):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (_, start, end), label in zip(tokens, predictions):
        if label == "O":
            if current_label is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )
                current_label = None
            continue

        if label == current_label and start <= current_end:
            current_end = end
        else:
            if current_label is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )

            current_label = label
            current_start = start
            current_end = end

    if current_label is not None:
        spans.append(
            (current_start, current_end, current_label)
        )

    return spans


def spans_from_bio_predictions(tokens, predictions):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (_, start, end), tag in zip(tokens, predictions):
        if tag == "O":
            if current_label is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )
                current_label = None
            continue

        if "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix, label = "B", tag

        if (
            prefix == "I"
            and label == current_label
            and start <= current_end
        ):
            current_end = end
        else:
            if current_label is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )

            current_label = label
            current_start = start
            current_end = end

    if current_label is not None:
        spans.append(
            (current_start, current_end, current_label)
        )

    return spans


def gold_spans(record):
    return {
        (
            int(entity["start"]),
            int(entity["end"]),
            entity["label"].upper()
        )
        for entity in record.get("entities", [])
    }


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(all_gold, all_pred):
    correct = 0
    predicted = 0
    actual = 0

    for gold, pred in zip(all_gold, all_pred):
        correct += len(gold & pred)
        predicted += len(pred)
        actual += len(gold)

    precision = correct / predicted if predicted else 0.0
    recall = correct / actual if actual else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


def token_labels_for_record(record, predicted_spans):
    tokens, gold_bio = bio_from_entities(
        record["text"],
        record.get("entities", [])
    )

    gold_labels = [
        tag.split("-", 1)[-1] if "-" in tag else tag
        for tag in gold_bio
    ]

    predicted_labels = []
    ordered_spans = sorted(
        predicted_spans,
        key=lambda item: (item[0], item[1])
    )

    for _, start, end in tokens:
        label = "O"

        for ps, pe, pl in ordered_spans:
            if end > ps and start < pe:
                label = str(pl).upper()
                break

        predicted_labels.append(label)

    return tokens, gold_labels, predicted_labels


def calculate_token_metrics(records, all_pred):
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for record, predicted_spans in zip(records, all_pred):
        _, gold_labels, predicted_labels = token_labels_for_record(
            record,
            predicted_spans
        )

        for gold, pred in zip(gold_labels, predicted_labels):
            total += 1

            if gold == pred:
                correct += 1

            if gold != "O" and pred == gold:
                true_positive += 1
            elif gold == "O" and pred != "O":
                false_positive += 1
            elif gold != "O" and pred != gold:
                false_negative += 1
                if pred != "O":
                    false_positive += 1

    accuracy = correct / total if total else 0.0

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return accuracy, precision, recall, f1


def evaluate_predictions(name, records, predictions):
    all_gold = [gold_spans(record) for record in records]

    entity_precision, entity_recall, entity_f1 = calculate_metrics(
        all_gold,
        predictions
    )

    token_acc, token_precision, token_recall, token_f1 = (
        calculate_token_metrics(records, predictions)
    )

    gold_count = sum(len(items) for items in all_gold)
    predicted_count = sum(len(items) for items in predictions)

    exact_count = sum(
        len(gold & pred)
        for gold, pred in zip(all_gold, predictions)
    )

    result = {
        "Model": name,
        "Token Accuracy": token_acc,
        "Token Precision": token_precision,
        "Token Recall": token_recall,
        "Token F1-score": token_f1,
        "Entity Precision": entity_precision,
        "Entity Recall": entity_recall,
        "Entity F1-score": entity_f1,
        "Gold Entities": gold_count,
        "Predicted Entities": predicted_count,
        "Exact Entity Matches": exact_count
    }

    print("\n", name)
    print("Token Accuracy:", round(token_acc, 4))
    print("Token Precision:", round(token_precision, 4))
    print("Token Recall:", round(token_recall, 4))
    print("Token F1-score:", round(token_f1, 4))
    print("Entity Precision:", round(entity_precision, 4))
    print("Entity Recall:", round(entity_recall, 4))
    print("Entity F1-score:", round(entity_f1, 4))
    print(
        "Gold={}, Predicted={}, Exact={}".format(
            gold_count, predicted_count, exact_count
        )
    )

    return result


# ==================================================
# DECISION TREE / RANDOM FOREST
# ==================================================

def shape(w):
    return {
        "is_upper": w.isupper(),
        "is_title": w.istitle(),
        "is_digit": w.isdigit(),
        "has_digit": any(c.isdigit() for c in w),
        "has_at": "@" in w,
        "has_slash": "/" in w,
        "len": min(len(w), 20),
        "prefix2": w[:2].lower(),
        "prefix3": w[:3].lower(),
        "suffix2": w[-2:].lower(),
        "suffix3": w[-3:].lower()
    }


def feature_dict(words, i):
    w = words[i]
    d = shape(w)
    d["word"] = w.lower()

    d["prev1"] = words[i - 1].lower() if i > 0 else "<START>"
    d["next1"] = (
        words[i + 1].lower()
        if i + 1 < len(words) else "<END>"
    )
    d["prev2"] = words[i - 2].lower() if i > 1 else "<START>"
    d["next2"] = (
        words[i + 2].lower()
        if i + 2 < len(words) else "<END>"
    )

    return d


def predict_tree_model(records, model, vectorizer):
    predictions = []

    for record in records:
        tokens = tokenize_with_offsets(record["text"])
        words = [token[0] for token in tokens]

        features = [
            feature_dict(words, i)
            for i in range(len(words))
        ]

        if not features:
            predictions.append(set())
            continue

        X = vectorizer.transform(features)
        labels = model.predict(X)

        spans = spans_from_class_predictions(tokens, labels)
        predictions.append(set(spans))

    return predictions


# ==================================================
# CNN MODEL
# ==================================================

class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=64, channels=96):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, emb, padding_idx=0
        )

        self.conv = nn.Conv1d(
            emb, channels, kernel_size=3, padding=1
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        z = self.embedding(x).transpose(1, 2)
        z = torch.relu(self.conv(z))
        z = self.pool(z).squeeze(-1)
        return self.fc(z)


def predict_cnn(records):
    vocab_path = MODEL_DIR / "cnn_bio_vocab.joblib"
    labels_path = MODEL_DIR / "cnn_bio_labels.json"
    weights_path = MODEL_DIR / "cnn_bio_model.pt"

    for path in (vocab_path, labels_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(
                "Missing CNN model file: {}".format(path)
            )

    vocab = joblib.load(vocab_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        bio_labels = json.load(f)

    model = TextCNN(
        len(vocab), len(bio_labels)
    ).to(DEVICE)

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    predictions = []

    with torch.no_grad():
        for record in records:
            tokens = tokenize_with_offsets(record["text"])

            if not tokens:
                predictions.append(set())
                continue

            words = [token[0].lower() for token in tokens]
            ids = [
                vocab.get(word, vocab["<UNK>"])
                for word in words
            ]

            predicted_tags = []

            for i in range(len(ids)):
                context = []

                for j in range(i - 2, i + 3):
                    if 0 <= j < len(ids):
                        context.append(ids[j])
                    else:
                        context.append(vocab["<PAD>"])

                x = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=DEVICE
                )

                output = model(x)
                pred_id = int(output.argmax(dim=1).item())
                predicted_tags.append(bio_labels[pred_id])

            spans = spans_from_bio_predictions(
                tokens, predicted_tags
            )

            predictions.append(set(spans))

    return predictions


# ==================================================
# BiLSTM-CRF MODEL
# ==================================================

EMBED_DIM = 100
HIDDEN_DIM = 128


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()

        self.num_tags = num_tags
        self.transitions = nn.Parameter(
            torch.zeros(num_tags, num_tags)
        )
        self.start_transitions = nn.Parameter(
            torch.zeros(num_tags)
        )
        self.end_transitions = nn.Parameter(
            torch.zeros(num_tags)
        )

    def decode(self, emissions, mask):
        results = []

        for b in range(emissions.size(0)):
            length = int(mask[b].sum().item())

            if length == 0:
                results.append([])
                continue

            score = self.start_transitions + emissions[b, 0]
            history = []

            for t in range(1, length):
                candidates = (
                    score.unsqueeze(1) + self.transitions
                )

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

        self.embedding = nn.Embedding(
            vocab_size, EMBED_DIM, padding_idx=0
        )

        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(HIDDEN_DIM, num_tags)
        self.crf = LinearChainCRF(num_tags)

    def emissions(self, token_ids):
        embedded = self.embedding(token_ids)
        encoded, _ = self.lstm(embedded)

        return self.classifier(self.dropout(encoded))

    def predict(self, token_ids, mask):
        return self.crf.decode(
            self.emissions(token_ids), mask
        )


def predict_bilstm(records):
    model_dir = MODEL_DIR / "bilstm_crf"

    vocab_path = model_dir / "vocab.json"
    labels_path = model_dir / "labels.json"
    weights_path = model_dir / "model.pt"

    for path in (vocab_path, labels_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(
                "Missing BiLSTM-CRF file: {}".format(path)
            )

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(labels_path, "r", encoding="utf-8") as f:
        bio_labels = json.load(f)

    model = BiLSTMCRF(
        len(vocab), len(bio_labels)
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE)
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for record in records:
            tokens = tokenize_with_offsets(record["text"])

            if not tokens:
                predictions.append(set())
                continue

            words = [token[0].lower() for token in tokens]
            ids = [
                vocab.get(word, vocab["<UNK>"])
                for word in words
            ]

            x = torch.tensor(
                [ids],
                dtype=torch.long,
                device=DEVICE
            )

            mask = x != 0
            paths = model.predict(x, mask)[0]
            tags = [bio_labels[tag] for tag in paths]

            spans = spans_from_bio_predictions(tokens, tags)
            predictions.append(set(spans))

    return predictions


# ==================================================
# BIOCLINICALBERT
# ==================================================

def predict_bioclinicalbert(records):
    model_dir = MODEL_DIR / "bioclinicalbert"
    labels_path = model_dir / "bio_labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(
            "BioClinicalBERT labels file is missing."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        bio_labels = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir)
    ).to(DEVICE)

    model.eval()
    predictions = []

    with torch.no_grad():
        for record in records:
            text = record["text"]

            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            offsets = encoding.pop("offset_mapping")[0].tolist()

            inputs = {
                key: value.to(DEVICE)
                for key, value in encoding.items()
            }

            logits = model(**inputs).logits
            predicted_ids = logits.argmax(dim=-1)[0].tolist()

            valid_tokens = []
            valid_tags = []

            for offset, pred_id in zip(offsets, predicted_ids):
                start, end = offset

                if start == end:
                    continue

                valid_tokens.append(
                    (text[start:end], start, end)
                )

                valid_tags.append(bio_labels[pred_id])

            spans = spans_from_bio_predictions(
                valid_tokens, valid_tags
            )

            predictions.append(set(spans))

    return predictions


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(results):
    output_path = RESULT_DIR / "model_comparison.csv"

    fieldnames = [
        "Model",
        "Token Accuracy",
        "Token Precision",
        "Token Recall",
        "Token F1-score",
        "Entity Precision",
        "Entity Recall",
        "Entity F1-score",
        "Gold Entities",
        "Predicted Entities",
        "Exact Entity Matches"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", output_path)


# ==================================================
# MAIN
# ==================================================

def main():
    print("=" * 65)
    print("MEDICAL RECORD DE-IDENTIFICATION")
    print("MODEL EVALUATION")
    print("=" * 65)

    print("\nDevice:", DEVICE)
    print("\nLoading test dataset...")

    records = load_jsonl(TEST_PATH)
    print("Test records:", len(records))

    results = []

    print("\nEvaluating Decision Tree...")

    vectorizer = joblib.load(
        MODEL_DIR / "feature_vectorizer.joblib"
    )

    dt = joblib.load(MODEL_DIR / "decision_tree.joblib")

    dt_predictions = predict_tree_model(
        records, dt, vectorizer
    )

    results.append(
        evaluate_predictions(
            "Decision Tree", records, dt_predictions
        )
    )

    print("\nEvaluating Random Forest...")

    rf = joblib.load(MODEL_DIR / "random_forest.joblib")

    rf_predictions = predict_tree_model(
        records, rf, vectorizer
    )

    results.append(
        evaluate_predictions(
            "Random Forest", records, rf_predictions
        )
    )

    print("\nEvaluating CNN...")

    cnn_predictions = predict_cnn(records)

    results.append(
        evaluate_predictions(
            "CNN", records, cnn_predictions
        )
    )

    print("\nEvaluating BiLSTM-CRF...")

    bilstm_predictions = predict_bilstm(records)

    results.append(
        evaluate_predictions(
            "BiLSTM-CRF", records, bilstm_predictions
        )
    )

    print("\nEvaluating BioClinicalBERT...")

    bert_predictions = predict_bioclinicalbert(records)

    results.append(
        evaluate_predictions(
            "BioClinicalBERT", records, bert_predictions
        )
    )

    print("\n" + "=" * 100)
    print("ENTITY-LEVEL MODEL COMPARISON")
    print("=" * 100)

    print(
        "{:<20} {:<15} {:<15} {:<15}".format(
            "Model",
            "Precision",
            "Recall",
            "F1-score"
        )
    )

    print("-" * 70)

    for result in results:
        print(
            "{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                result["Model"],
                result["Entity Precision"],
                result["Entity Recall"],
                result["Entity F1-score"]
            )
        )

    save_results(results)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()