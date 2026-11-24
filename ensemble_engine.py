
import re
import hashlib
import joblib
import torch
import numpy as np

from pathlib import Path
from common import LABELS, tokenize_with_offsets
from train_models import TextCNN

MODEL_DIR = Path("models")


class EnsembleEngine:

    def __init__(self):
        self.vec = joblib.load(
            MODEL_DIR / "feature_vectorizer.joblib"
        )

        self.dt = joblib.load(
            MODEL_DIR / "decision_tree.joblib"
        )

        self.rf = joblib.load(
            MODEL_DIR / "random_forest.joblib"
        )

        self.vocab = joblib.load(
            MODEL_DIR / "cnn_vocab.joblib"
        )

        self.cnn = TextCNN(
            len(self.vocab),
            len(LABELS)
        )

        self.cnn.load_state_dict(
            torch.load(
                MODEL_DIR / "cnn_model.pt",
                map_location="cpu",
                weights_only=True
            )
        )

        self.cnn.eval()

    def regex_labels(self, text):
        spans = []

        patterns = [
            (
                "EMAIL",
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),
            (
                "PHONE",
                r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"
            ),
            (
                "DATE",
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            ),
            (
                "ID",
                r"\bMRN[- ]?\d{4,10}\b"
            )
        ]

        for label, pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                spans.append((
                    match.start(),
                    match.end(),
                    label,
                    1.0
                ))

        return spans

    def _tree_probs(self, words):
        features = []

        for i, word in enumerate(words):
            feature = {
                "word": word.lower(),
                "prev1": words[i - 1].lower() if i > 0 else "<START>",
                "next1": words[i + 1].lower() if i + 1 < len(words) else "<END>",
                "prev2": words[i - 2].lower() if i > 1 else "<START>",
                "next2": words[i + 2].lower() if i + 2 < len(words) else "<END>",
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

            features.append(feature)

        X = self.vec.transform(features)

        return (
            self.dt.predict_proba(X),
            self.rf.predict_proba(X),
            self.dt.classes_,
            self.rf.classes_
        )

    def predict(self, text):
        if not text:
            return []

        tokens = tokenize_with_offsets(text)

        if not tokens:
            return []

        words = [token[0] for token in tokens]

        dtp, rfp, dt_classes, rf_classes = self._tree_probs(words)

        half_window = 2
        cnn_inputs = []

        for i, word in enumerate(words):
            ids = []

            for j in range(i - half_window, i + half_window + 1):
                if 0 <= j < len(words):
                    token_id = self.vocab.get(
                        words[j].lower(),
                        self.vocab["<UNK>"]
                    )
                else:
                    token_id = self.vocab["<PAD>"]

                ids.append(token_id)

            cnn_inputs.append(ids)

        with torch.no_grad():
            inputs = torch.tensor(
                cnn_inputs,
                dtype=torch.long
            )

            cnn_probs = torch.softmax(
                self.cnn(inputs),
                dim=1
            ).cpu().numpy()

        predictions = []

        for i, (word, start, end) in enumerate(tokens):
            scores = {label: 0.0 for label in LABELS}

            for j, label in enumerate(LABELS):
                if j < cnn_probs.shape[1]:
                    scores[label] += 0.40 * float(cnn_probs[i, j])

            for j, label in enumerate(dt_classes):
                scores[label] += 0.25 * float(dtp[i, j])

            for j, label in enumerate(rf_classes):
                scores[label] += 0.35 * float(rfp[i, j])

            label = max(scores, key=scores.get)

            if label != "O":
                predictions.append((
                    start,
                    end,
                    label,
                    scores[label]
                ))

        # Merge adjacent tokens belonging to the same entity type.
        spans = []

        for start, end, label, confidence in predictions:
            entity_type = label.split("-", 1)[-1]

            if (
                spans
                and spans[-1][2].split("-", 1)[-1] == entity_type
                and start <= spans[-1][1] + 1
            ):
                prev_start, prev_end, prev_label, prev_conf = spans[-1]

                spans[-1] = (
                    prev_start,
                    end,
                    prev_label,
                    max(prev_conf, confidence)
                )
            else:
                spans.append((
                    start,
                    end,
                    label,
                    confidence
                ))

        # Add regex detections.
        regex_spans = self.regex_labels(text)

        combined = [
            (start, end, label, confidence, False)
            for start, end, label, confidence in spans
        ]

        combined.extend(
            (start, end, label, confidence, True)
            for start, end, label, confidence in regex_spans
        )

        # Regex detections take priority over overlapping model spans.
        combined.sort(
            key=lambda item: (
                item[4],
                item[3],
                item[1] - item[0]
            ),
            reverse=True
        )

        chosen = []

        for span in combined:
            start, end, label, confidence, is_regex = span

            overlaps = any(
                not (end <= existing[0] or start >= existing[1])
                for existing in chosen
            )

            if not overlaps:
                chosen.append((
                    start,
                    end,
                    label,
                    confidence
                ))

        return sorted(chosen, key=lambda span: span[0])

    def mask(self, text, mode="mask"):
        spans = self.predict(text)
        output = text

        for start, end, label, confidence in sorted(
            spans,
            key=lambda span: span[0],
            reverse=True
        ):
            if mode == "mask":
                replacement = f"[{label.split('-', 1)[-1]}]"
            else:
                entity_type = label.split("-", 1)[-1]

                # Stable pseudonym for the same text and entity type.
                digest = hashlib.sha256(
                    f"{entity_type}:{text[start:end].casefold()}".encode("utf-8")
                ).hexdigest()[:8]

                replacement = f"{entity_type}_{digest}"

            output = output[:start] + replacement + output[end:]

        return output, spans
    
    