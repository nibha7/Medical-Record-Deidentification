
from pathlib import Path

from common import load_jsonl
from evaluate_models import (
    predict_cnn,
    predict_bilstm,
    predict_bioclinicalbert,
    predict_tree_model
)

import joblib


TEST_PATH = Path("data/test.jsonl")
MODEL_DIR = Path("models")


def get_gold(record):

    return {
        (
            int(e["start"]),
            int(e["end"]),
            e["label"].upper()
        )
        for e in record.get("entities", [])
    }


def show_span(text, span):

    start, end, label = span

    actual_text = text[start:end]

    return (
        f"{label}: '{actual_text}' "
        f"[{start}:{end}]"
    )


def diagnose_model(name, records, predictions):

    print("\n" + "=" * 75)
    print("DIAGNOSTIC:", name)
    print("=" * 75)

    total_gold = 0
    total_pred = 0
    total_correct = 0

    mismatch_count = 0

    for index, (record, predicted) in enumerate(
        zip(records, predictions)
    ):

        gold = get_gold(record)

        total_gold += len(gold)
        total_pred += len(predicted)

        correct = gold & predicted

        total_correct += len(correct)

        missing = gold - predicted
        extra = predicted - gold

        if missing or extra:

            mismatch_count += 1

            if mismatch_count > 5:
                continue

            print("\nRECORD:", index)

            print("\nTEXT:")
            print(record["text"])

            print("\nCORRECT PREDICTIONS:")

            for span in sorted(correct):
                print("  ", show_span(record["text"], span))

            print("\nMISSING GOLD ENTITIES:")

            for span in sorted(missing):
                print("  ", show_span(record["text"], span))

            print("\nEXTRA PREDICTED ENTITIES:")

            for span in sorted(extra):
                print("  ", show_span(record["text"], span))

    print("\nSUMMARY")

    print("Gold entities:", total_gold)
    print("Predicted entities:", total_pred)
    print("Exact matches:", total_correct)
    print("Records with mismatches:", mismatch_count)


def main():

    print("Loading test data...")

    records = load_jsonl(TEST_PATH)

    print("Records:", len(records))

    # Decision Tree

    print("\nLoading Decision Tree...")

    vectorizer = joblib.load(
        MODEL_DIR / "feature_vectorizer.joblib"
    )

    dt = joblib.load(
        MODEL_DIR / "decision_tree.joblib"
    )

    dt_predictions = predict_tree_model(
        records,
        dt,
        vectorizer
    )

    diagnose_model(
        "Decision Tree",
        records,
        dt_predictions
    )

    # Random Forest

    print("\nLoading Random Forest...")

    rf = joblib.load(
        MODEL_DIR / "random_forest.joblib"
    )

    rf_predictions = predict_tree_model(
        records,
        rf,
        vectorizer
    )

    diagnose_model(
        "Random Forest",
        records,
        rf_predictions
    )

    # CNN

    print("\nLoading CNN...")

    cnn_predictions = predict_cnn(records)

    diagnose_model(
        "CNN",
        records,
        cnn_predictions
    )

    # BiLSTM-CRF

    print("\nLoading BiLSTM-CRF...")

    bilstm_predictions = predict_bilstm(records)

    diagnose_model(
        "BiLSTM-CRF",
        records,
        bilstm_predictions
    )

    # BioClinicalBERT

    print("\nLoading BioClinicalBERT...")

    bert_predictions = predict_bioclinicalbert(records)

    diagnose_model(
        "BioClinicalBERT",
        records,
        bert_predictions
    )


if __name__ == "__main__":

    main()

