
import json
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

TEST_FILE = "data/test.jsonl"

def load_test_data():
    texts = []
    labels = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["entities"])

    return texts, labels


def evaluate_weak_baseline(y_true, y_pred):

    precision = precision_score(
        y_true, y_pred,
        average="micro",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="micro",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="micro",
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    print("\nWEAK BASELINE MODEL RESULTS")
    print("---------------------------")
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")


if __name__ == "__main__":
    print("Weak baseline evaluation")
    print("Please connect the dataset's actual token labels")
    print("and predictions before running the evaluation.")
