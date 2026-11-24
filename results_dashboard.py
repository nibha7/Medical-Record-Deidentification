
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


RESULTS_DIR = Path("results")

st.set_page_config(
    page_title="MedDeID ML Results",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MedDeID — Machine Learning Performance")
st.caption(
    "Evaluation of CNN, Decision Tree, Random Forest, and Ensemble"
)

metrics_path = RESULTS_DIR / "metrics.csv"

if not metrics_path.exists():
    st.warning(
        "Evaluation results were not found. Run evaluate.py first."
    )
    st.stop()

metrics = pd.read_csv(metrics_path)

required_columns = {
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
}

if not required_columns.issubset(metrics.columns):
    st.error("The metrics file is missing required columns.")
    st.stop()

# Show metrics as percentages.
ensemble_row = metrics[metrics["Model"] == "Ensemble"]

if not ensemble_row.empty:
    row = ensemble_row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{row['Accuracy']:.2%}")
    c2.metric("Precision", f"{row['Precision']:.2%}")
    c3.metric("Recall", f"{row['Recall']:.2%}")
    c4.metric("F1-Score", f"{row['F1-Score']:.2%}")

st.divider()

# --------------------------------------------------
# GRAPH 1: MODEL COMPARISON
# --------------------------------------------------

st.subheader("1. Model Performance Comparison")

metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

chart_data = metrics.set_index("Model")[metric_names]

fig, ax = plt.subplots(figsize=(11, 5))

chart_data.plot(
    kind="bar",
    ax=ax,
    color=["#4285F4", "#34A853", "#FB8C00", "#AB47BC"],
    width=0.78
)

ax.set_title("CNN vs Decision Tree vs Random Forest vs Ensemble")
ax.set_ylabel("Score (0 to 1)")
ax.set_xlabel("Machine Learning Model")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="lower right")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    metrics.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
    }),
    use_container_width=True
)

st.download_button(
    "Download model metrics CSV",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="model_metrics.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# GRAPH 2: ENSEMBLE CONFUSION MATRIX
# --------------------------------------------------

st.subheader("2. Ensemble Confusion Matrix")

cm_path = RESULTS_DIR / "ensemble_confusion_matrix.csv"

if cm_path.exists():
    cm_df = pd.read_csv(cm_path, index_col=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        cm_df.values,
        interpolation="nearest",
        cmap="YlGnBu"
    )

    ax.set_title("Ensemble — Actual vs Predicted Labels")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)

    threshold = cm_df.values.max() / 2 if cm_df.size else 0

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = cm_df.iloc[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8
            )

    fig.colorbar(image, ax=ax, label="Number of Tokens")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Confusion matrix is not available yet.")

st.divider()

# --------------------------------------------------
# GRAPH 3: PERFORMANCE BY ENTITY LABEL
# --------------------------------------------------

st.subheader("3. Ensemble Performance by PHI Category")

report_path = RESULTS_DIR / "ensemble_classification_report.csv"

if report_path.exists():
    report_df = pd.read_csv(report_path, index_col=0)

    required = {"precision", "recall", "f1-score"}

    if required.issubset(report_df.columns):
        entity_names = [
            label for label in report_df.index
            if label not in [
                "O", "accuracy", "macro avg", "weighted avg"
            ]
        ]

        entity_df = report_df.loc[
            report_df.index.isin(entity_names),
            ["precision", "recall", "f1-score"]
        ]

        if not entity_df.empty:
            fig, ax = plt.subplots(figsize=(11, 5))

            entity_df.plot(
                kind="bar",
                ax=ax,
                color=["#00ACC1", "#FF7043", "#7E57C2"]
            )

            ax.set_title("Precision, Recall and F1-Score by PHI Label")
            ax.set_ylabel("Score (0 to 1)")
            ax.set_xlabel("PHI Label")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(loc="lower right")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No entity-category metrics were found.")
else:
    st.info("Classification report is not available yet.")

st.caption(
    "Metrics are calculated from the configured test set. "
    "High scores do not by themselves establish performance "
    "on real-world medical records."
)