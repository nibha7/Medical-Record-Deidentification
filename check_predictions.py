
import json
import torch
import joblib

from common import load_jsonl, tokenize_with_offsets, LABELS

from evaluate_models import (
    predict_cnn,
    predict_bilstm,
    predict_bioclinicalbert,
    predict_tree_model
)

records = load_jsonl("data/test.jsonl")

print("Testing prediction differences...")
print("Number of test records:", len(records))

cnn = predict_cnn(records)
bilstm = predict_bilstm(records)
bert = predict_bioclinicalbert(records)

vectorizer = joblib.load("models/feature_vectorizer.joblib")
dt = joblib.load("models/decision_tree.joblib")
rf = joblib.load("models/random_forest.joblib")

tree = predict_tree_model(records, dt, vectorizer)
forest = predict_tree_model(records, rf, vectorizer)

models = {
    "Decision Tree": tree,
    "Random Forest": forest,
    "CNN": cnn,
    "BiLSTM-CRF": bilstm,
    "BioClinicalBERT": bert
}

for name, predictions in models.items():
    count = sum(len(x) for x in predictions)
    print(name, "Predicted entities:", count)

print("\nComparing model predictions:")

names = list(models.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):

        a = models[names[i]]
        b = models[names[j]]

        identical = sum(
            x == y for x, y in zip(a, b)
        )

        print(
            names[i], "vs", names[j],
            "| Identical records:",
            identical, "/", len(records)
        )
