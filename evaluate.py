
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from common import (
    load_jsonl,
    bio_from_entities,
    tokenize_with_offsets,
    LABELS,
    LABEL2ID,
)

from train_models import TextCNN, feature_dict
from ensemble_engine import EnsembleEngine


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TEST_FILE = DATA_DIR / "test.jsonl"
DEVICE = torch.device("cpu")

PHI_LABELS = [label for label in LABELS if label != "O"]
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

METRIC_COLUMNS = [
    "F1-Score",
    "Recall",
    "Precision",
    "Accuracy",
]


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

def load_models():
    vectorizer = joblib.load(
        MODEL_DIR / "feature_vectorizer.joblib"
    )

    decision_tree = joblib.load(
        MODEL_DIR / "decision_tree.joblib"
    )

    random_forest = joblib.load(
        MODEL_DIR / "random_forest.joblib"
    )

    vocab = joblib.load(
        MODEL_DIR / "cnn_vocab.joblib"
    )

    cnn = TextCNN(
        len(vocab),
        len(LABELS),
    )

    state = torch.load(
        MODEL_DIR / "cnn_model.pt",
        map_location=DEVICE,
        weights_only=True,
    )

    cnn.load_state_dict(state)
    cnn.eval()

    return (
        vectorizer,
        decision_tree,
        random_forest,
        vocab,
        cnn,
    )


# --------------------------------------------------
# CNN PREDICTIONS
# --------------------------------------------------

def predict_cnn(text, vocab, cnn):
    tokens = tokenize_with_offsets(text)

    if not tokens:
        return []

    words = [token[0].lower() for token in tokens]
    half_window = 2
    inputs = []

    for i in range(len(words)):
        window = []

        for j in range(
            i - half_window,
            i + half_window + 1,
        ):
            if 0 <= j < len(words):
                token_id = vocab.get(
                    words[j],
                    vocab["<UNK>"],
                )
            else:
                token_id = vocab["<PAD>"]

            window.append(token_id)

        inputs.append(window)

    with torch.no_grad():
        x = torch.tensor(
            inputs,
            dtype=torch.long,
        )

        probabilities = torch.softmax(
            cnn(x),
            dim=1,
        )

        predicted_ids = (
            probabilities.argmax(dim=1).tolist()
        )

    return [
        ID2LABEL[index]
        for index in predicted_ids
    ]


# --------------------------------------------------
# DECISION TREE AND RANDOM FOREST
# --------------------------------------------------

def predict_tree_models(
    text,
    vectorizer,
    decision_tree,
    random_forest,
):
    tokens = tokenize_with_offsets(text)
    words = [token[0] for token in tokens]

    if not words:
        return [], []

    features = [
        feature_dict(words, i)
        for i in range(len(words))
    ]

    X = vectorizer.transform(features)

    dt_predictions = (
        decision_tree.predict(X).tolist()
    )

    rf_predictions = (
        random_forest.predict(X).tolist()
    )

    return dt_predictions, rf_predictions


# --------------------------------------------------
# CONVERT TOKEN PREDICTIONS TO MASKED TEXT
# --------------------------------------------------

def mask_token_predictions(
    text,
    tokens,
    predictions,
):
    spans = []
    i = 0

    while i < len(tokens):
        label = predictions[i]

        if label == "O":
            i += 1
            continue

        start = tokens[i][1]
        end = tokens[i][2]
        j = i + 1

        # Merge consecutive tokens with the same label.
        while (
            j < len(tokens)
            and predictions[j] == label
        ):
            end = tokens[j][2]
            j += 1

        spans.append((start, end, label))
        i = j

    output = text

    # Replace from right to left to preserve offsets.
    for start, end, label in reversed(spans):
        output = (
            output[:start]
            + f"[{label}]"
            + output[end:]
        )

    return output


def mask_ensemble_predictions(text, spans):
    output = text

    for start, end, label, confidence in sorted(
        spans,
        key=lambda item: item[0],
        reverse=True,
    ):
        label = label.split("-", 1)[-1]

        output = (
            output[:start]
            + f"[{label}]"
            + output[end:]
        )

    return output


# --------------------------------------------------
# CONVERT ENSEMBLE SPANS TO TOKEN LABELS
# --------------------------------------------------

def spans_to_token_labels(tokens, spans):
    predictions = ["O"] * len(tokens)

    for start, end, label, *_ in spans:
        label = label.split("-", 1)[-1]

        for i, (_, token_start, token_end) in enumerate(
            tokens
        ):
            if (
                token_end > start
                and token_start < end
            ):
                predictions[i] = label

    return predictions


# --------------------------------------------------
# GOLD LABELS
# --------------------------------------------------

def flatten_gold_labels(text, entities):
    _, bio_labels = bio_from_entities(
        text,
        entities,
    )

    return [
        label.split("-", 1)[-1]
        for label in bio_labels
    ]


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        # PHI micro metrics exclude the O label.
        "Precision": precision_score(
            y_true,
            y_pred,
            labels=PHI_LABELS,
            average="micro",
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            labels=PHI_LABELS,
            average="micro",
            zero_division=0,
        ),

        "F1-Score": f1_score(
            y_true,
            y_pred,
            labels=PHI_LABELS,
            average="micro",
            zero_division=0,
        ),
    }


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    print("Loading test dataset...")
    records = load_jsonl(str(TEST_FILE))

    if not records:
        raise ValueError(
            "The test dataset is empty."
        )

    print("Loading trained models...")

    (
        vectorizer,
        decision_tree,
        random_forest,
        vocab,
        cnn,
    ) = load_models()

    ensemble = EnsembleEngine()

    model_names = [
        "CNN",
        "Decision Tree",
        "Random Forest",
        "Ensemble",
    ]

    all_true = []
    all_predictions = {
        name: []
        for name in model_names
    }

    output_rows = []

    print(
        f"Evaluating {len(records)} test records..."
    )

    # Evaluate each test record with all four models.
    for record_number, record in enumerate(
        records,
        start=1,
    ):
        text = record["text"]
        entities = record["entities"]

        tokens = tokenize_with_offsets(text)
        gold = flatten_gold_labels(
            text,
            entities,
        )

        cnn_pred = predict_cnn(
            text,
            vocab,
            cnn,
        )

        dt_pred, rf_pred = predict_tree_models(
            text,
            vectorizer,
            decision_tree,
            random_forest,
        )

        ensemble_spans = ensemble.predict(text)

        ensemble_pred = spans_to_token_labels(
            tokens,
            ensemble_spans,
        )

        predictions = {
            "CNN": cnn_pred,
            "Decision Tree": dt_pred,
            "Random Forest": rf_pred,
            "Ensemble": ensemble_pred,
        }

        # Validate prediction lengths.
        for name, pred in predictions.items():
            if len(pred) != len(gold):
                raise ValueError(
                    f"Token count mismatch in record "
                    f"{record_number} for {name}: "
                    f"{len(pred)} predictions vs "
                    f"{len(gold)} gold labels."
                )

        all_true.extend(gold)

        for name in model_names:
            all_predictions[name].extend(
                predictions[name]
            )

        # Save the original record and all model outputs.
        output_rows.append({
            "record_number": record_number,
            "original_text": text,

            "correct_labels": json.dumps(gold),
            "cnn_labels": json.dumps(cnn_pred),
            "decision_tree_labels": json.dumps(dt_pred),
            "random_forest_labels": json.dumps(rf_pred),
            "ensemble_labels": json.dumps(ensemble_pred),

            "cnn_output": mask_token_predictions(
                text,
                tokens,
                cnn_pred,
            ),

            "decision_tree_output": mask_token_predictions(
                text,
                tokens,
                dt_pred,
            ),

            "random_forest_output": mask_token_predictions(
                text,
                tokens,
                rf_pred,
            ),

            "ensemble_output": mask_ensemble_predictions(
                text,
                ensemble_spans,
            ),
        })

        if record_number % 500 == 0:
            print(
                f"Processed {record_number} records..."
            )

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    rows = []

    for name in model_names:
        metrics = calculate_metrics(
            all_true,
            all_predictions[name],
        )

        rows.append({
            "Model": name,
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)

    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------
    # CLASSIFICATION REPORTS
    # --------------------------------------------------

    for name in model_names:
        report = classification_report(
            all_true,
            all_predictions[name],
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        )

        report_df = pd.DataFrame(report).transpose()

        safe_name = name.lower().replace(" ", "_")

        report_df.to_csv(
            RESULTS_DIR / f"{safe_name}_report.csv"
        )

    # --------------------------------------------------
    # SAVE ALL MODEL OUTPUTS
    # --------------------------------------------------

    output_path = RESULTS_DIR / "model_outputs.csv"

    pd.DataFrame(output_rows).to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------

    print("\nMODEL COMPARISON")

    print(
        metrics_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-Score": "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------
    # RANK MODELS - FIXED VERSION
    # --------------------------------------------------

    ranking = metrics_df.sort_values(
        by=METRIC_COLUMNS,
        ascending=False,
    ).reset_index(drop=True)

    print("\nMODEL RANKING")

    print(
        ranking.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-Score": "{:.4f}".format,
            },
        )
    )

    # Convert the values to numeric NumPy arrays.
    # This avoids the pandas rounding error.
    scores = ranking[
        METRIC_COLUMNS
    ].astype(float).to_numpy()

    best_scores = scores[0]

    tied_models = [
        ranking.iloc[i]["Model"]
        for i in range(len(ranking))
        if np.allclose(
            scores[i],
            best_scores,
            rtol=0,
            atol=1e-10,
        )
    ]

    print("\nBEST MODEL RESULT")

    if len(tied_models) == 1:
        print(
            "Best by the selected metrics: "
            f"{tied_models[0]}"
        )
    else:
        print(
            "Tie between: "
            + ", ".join(tied_models)
            + ". More testing is needed to select "
            "a winner."
        )

    # --------------------------------------------------
    # SHOW FIRST RECORD'S MASKED OUTPUTS
    # --------------------------------------------------

    print(
        "\nFIRST TEST RECORD: MODEL OUTPUT COMPARISON"
    )

    if output_rows:
        first = output_rows[0]

        print("\nOriginal:")
        print(first["original_text"])

        for name, key in [
            ("CNN", "cnn_output"),
            ("Decision Tree", "decision_tree_output"),
            ("Random Forest", "random_forest_output"),
            ("Ensemble", "ensemble_output"),
        ]:
            print(f"\n{name}:")
            print(first[key])

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print("\nFiles saved successfully:")
    print(f"Comparison metrics: {metrics_path}")
    print(f"All model outputs: {output_path}")
    print(f"Classification reports: {RESULTS_DIR}")

    print(
        "\nNote: The highest metrics do not by themselves "
        "prove that a model is safe for real medical records. "
        "Validate with independent test records."
    )


if __name__ == "__main__":
    main()